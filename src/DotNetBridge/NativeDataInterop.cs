﻿//------------------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//------------------------------------------------------------------------------

using System;
using System.Collections.Generic;
using System.Linq;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Text;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using System.Buffers;

namespace Microsoft.ML.DotNetBridge
{
    public unsafe static partial class Bridge
    {
        const int UTF8_BUFFER_SIZE = 10 * 1024 * 1024; // 10 MB
        const int INDICES_BUFFER_SIZE = 1024 * 1024; // 1 Mln

        /// <summary>
        /// This is provided by the native code and represents a native data source. It provides schema
        /// information and call backs for iteration.
        /// </summary>
        [StructLayout(LayoutKind.Explicit)]
        private struct DataSourceBlock
        {
#pragma warning disable 649 // never assigned
            [FieldOffset(0x00)]
            public readonly long ccol;
            [FieldOffset(0x08)]
            public readonly long crow;
            [FieldOffset(0x10)]
            public readonly sbyte** names;
            [FieldOffset(0x18)]
            public readonly InternalDataKind* kinds;
            [FieldOffset(0x20)]
            public readonly long* keyCards;
            [FieldOffset(0x28)]
            public readonly long* vecCards;
            [FieldOffset(0x30)]
            // Call back pointers.
            public readonly void** getters;
            [FieldOffset(0x38)]
            public readonly void* labelsGetter;
#pragma warning restore 649 // never assigned
        }

        /// <summary>
        /// This is provided by the native code and represents a native data source. It provides schema
        /// information and call backs for iteration.
        /// </summary>
        [StructLayout(LayoutKind.Explicit)]
        private struct DataViewBlock
        {
            // Number of columns.
            [FieldOffset(0x00)]
            public long ccol;

            // Number of rows, zero for unknown.
            [FieldOffset(0x08)]
            public long crow;

            // The following fields are all "arrays" with length ccol.

            // Column names as null terminated UTF8 encoded strings.
            [FieldOffset(0x10)]
            public sbyte** names;

            // Column data kinds.
            [FieldOffset(0x18)]
            public InternalDataKind* kinds;

            // For columns that have key type, these contain the cardinalities of the
            // key types. Zero means unbounded, -1 means not a key type.
            [FieldOffset(0x20)]
            public int* keyCards;

            // The number of values in each row of a column.
            // A value count of 0 means that each row of the
            // column is variable length.
            [FieldOffset(0x28)]
            public byte* valueCounts;
        }

        private struct ColumnMetadataInfo
        {
            public bool Expand;
            public string[] SlotNames;
            public Dictionary<uint, ReadOnlyMemory<char>> KeyNameValues;

            public ColumnMetadataInfo(bool expand, string[] slotNames, Dictionary<uint, ReadOnlyMemory<char>> keyNameValues)
            {
                Expand = expand;
                SlotNames = slotNames;
                KeyNameValues = keyNameValues;
            }
        }

        private static unsafe void SendViewToNativeAsDataFrame(IChannel ch, EnvironmentBlock* penv, IDataView view, Dictionary<string, ColumnMetadataInfo> infos = null)
        {
            Contracts.AssertValue(ch);
            Contracts.Assert(penv != null);
            Contracts.AssertValue(view);
            Contracts.AssertValueOrNull(infos);
            if (penv->dataSink == null)
            {
                // Environment doesn't want any data!
                return;
            }

            var dataSink = MarshalDelegate<DataSink>(penv->dataSink);

            var schema = view.Schema;
            var colIndices = new List<int>(1000);
            var kindList = new ValueListBuilder<InternalDataKind>(INDICES_BUFFER_SIZE);
            var keyCardList = new ValueListBuilder<int>(INDICES_BUFFER_SIZE);
            var nameUtf8Bytes = new ValueListBuilder<byte>(UTF8_BUFFER_SIZE);
            var nameIndices = new ValueListBuilder<int>(INDICES_BUFFER_SIZE);
            var expandCols = new HashSet<int>(1000);
            var valueCounts = new List<byte>(1000);

            for (int col = 0; col < schema.Count; col++)
            {
                if (schema[col].IsHidden)
                    continue;

                var fullType = schema[col].Type;
                var itemType = fullType.GetItemType();
                var name = schema[col].Name;

                var kind = itemType.GetRawKind();
                int keyCard;

                byte valueCount = (fullType.GetValueCount() == 0) ? (byte)0 : (byte)1;

                if (itemType is KeyDataViewType)
                {
                    // Key types are returned as their signed counterparts in Python, so that -1 can be the missing value.
                    // For U1 and U2 kinds, we convert to a larger type to prevent overflow. For U4 and U8 kinds, we convert
                    // to I4 if the key count is known (since KeyCount is an I4), and to I8 otherwise.
                    switch (kind)
                    {
                    case InternalDataKind.U1:
                        kind = InternalDataKind.I2;
                        break;
                    case InternalDataKind.U2:
                        kind = InternalDataKind.I4;
                        break;
                    case InternalDataKind.U4:
                        // We convert known-cardinality U4 key types to I4.
                        kind = itemType.GetKeyCount() > 0 ? InternalDataKind.I4 : InternalDataKind.I8;
                        break;
                    case InternalDataKind.U8:
                        // We convert known-cardinality U8 key types to I4.
                        kind = itemType.GetKeyCount() > 0 ? InternalDataKind.I4 : InternalDataKind.I8;
                        break;
                    }

                    keyCard = itemType.GetKeyCountAsInt32();
                    if (!schema[col].HasKeyValues())
                        keyCard = -1;
                }
                else if (itemType.IsStandardScalar())
                {
                    switch (itemType.GetRawKind())
                    {
                    default:
                        throw Contracts.Except("Data type {0} not handled", itemType.GetRawKind());

                    case InternalDataKind.I1:
                    case InternalDataKind.I2:
                    case InternalDataKind.I4:
                    case InternalDataKind.I8:
                    case InternalDataKind.U1:
                    case InternalDataKind.U2:
                    case InternalDataKind.U4:
                    case InternalDataKind.U8:
                    case InternalDataKind.R4:
                    case InternalDataKind.R8:
                    case InternalDataKind.BL:
                    case InternalDataKind.TX:
                    case InternalDataKind.DT:
                        break;
                    }
                    keyCard = -1;
                }
                else
                {
                    throw Contracts.Except("Data type {0} not handled", itemType.GetRawKind());
                }

                int nSlots;
                ColumnMetadataInfo info;
                if (infos != null && infos.TryGetValue(name, out info) && info.Expand)
                {
                    expandCols.Add(col);
                    Contracts.Assert(fullType.IsKnownSizeVector());
                    nSlots = fullType.GetVectorSize();
                    if (info.SlotNames != null)
                    {
                        Contracts.Assert(info.SlotNames.Length == nSlots);
                        for (int i = 0; i < nSlots; i++)
                            AddUniqueName(info.SlotNames[i], ref nameIndices, ref nameUtf8Bytes);
                    }
                    else if (schema[col].HasSlotNames(nSlots))
                    {
                        var romNames = default(VBuffer<ReadOnlyMemory<char>>);
                        schema[col].Annotations.GetValue(AnnotationUtils.Kinds.SlotNames, ref romNames);
                        AddUniqueName(name, romNames, ref nameIndices, ref nameUtf8Bytes);
                    }
                    else
                    {
                        if (nSlots == 1)
                            AddUniqueName(name, ref nameIndices, ref nameUtf8Bytes);
                        else 
                            for (int i = 0; i < nSlots; i++)
                                AddUniqueName(name + "." + i, ref nameIndices, ref nameUtf8Bytes);
                    }
                }
                else
                {
                    nSlots = 1;
                    AddUniqueName(name, ref nameIndices, ref nameUtf8Bytes);
                }

                colIndices.Add(col);
                for (int i = 0; i < nSlots; i++)
                {
                    kindList.Append(kind);
                    keyCardList.Append(keyCard);
                    valueCounts.Add(valueCount);
                }
            }

            ch.Assert(kindList.Length == keyCardList.Length);
            ch.Assert(kindList.Length == nameIndices.Length);

            var kinds = kindList.AsSpan();
            var keyCards = keyCardList.AsSpan();
            var nameBytes = nameUtf8Bytes.AsSpan();
            var names = new byte*[nameIndices.Length];
            var valueCountsBytes = valueCounts.ToArray();

            fixed (InternalDataKind* prgkind = kinds)
            fixed (byte* prgbNames = nameBytes)
            fixed (byte** prgname = names)
            fixed (int* prgkeyCard = keyCards)
            fixed (byte* prgbValueCount = valueCountsBytes)
            {
                for (int iid = 0; iid < names.Length; iid++)
                    names[iid] = prgbNames + nameIndices[iid];

                DataViewBlock block;
                block.ccol = nameIndices.Length;
                block.crow = view.GetRowCount() ?? 0;
                block.names = (sbyte**)prgname;
                block.kinds = prgkind;
                block.keyCards = prgkeyCard;
                block.valueCounts = prgbValueCount;

                dataSink(penv, &block, out var setters, out var keyValueSetter);

                if (setters == null)
                {
                    // REVIEW: What should we do?
                    return;
                }
                ch.Assert(keyValueSetter != null);
                var kvSet = MarshalDelegate<KeyValueSetter>(keyValueSetter);
                using (var cursor = view.GetRowCursor(view.Schema.Where(col => colIndices.Contains(col.Index))))
                {
                    var fillers = new BufferFillerBase[colIndices.Count];
                    var pyColumn = 0;
                    var keyIndex = 0;
                    for (int i = 0; i < colIndices.Count; i++)
                    {
                        var type = schema[colIndices[i]].Type;
                        var itemType = type.GetItemType();
                        if ((itemType is KeyDataViewType) && schema[colIndices[i]].HasKeyValues())
                        {
                            ch.Assert(schema[colIndices[i]].HasKeyValues());
                            var keyValues = default(VBuffer<ReadOnlyMemory<char>>);
                            schema[colIndices[i]].Annotations.GetValue(AnnotationUtils.Kinds.KeyValues, ref keyValues);
                            for (int slot = 0; slot < type.GetValueCount(); slot++)
                            {
                                foreach (var kvp in keyValues.Items())
                                {
                                    if (kvp.Value.IsEmpty)
                                        kvSet(penv, keyIndex, kvp.Key, null, 0);
                                    else
                                    {
                                        byte[] bt = Encoding.UTF8.GetBytes(kvp.Value.ToString());
                                        fixed (byte* pt = bt)
                                            kvSet(penv, keyIndex, kvp.Key, (sbyte*)pt, bt.Length);
                                    }
                                }
                                keyIndex++;
                            }
                        }
                        fillers[i] = BufferFillerBase.Create(penv, cursor, pyColumn, colIndices[i], prgkind[pyColumn], type, setters[pyColumn]);

                        if ((type is VectorDataViewType) && (type.GetVectorSize() > 0))
                        {
                            pyColumn += type.GetVectorSize();
                        }
                        else pyColumn++;
                    }
                    for (int crow = 0; ; crow++)
                    {
                        // Advance to the next row.
                        if (!cursor.MoveNext())
                            break;

                        // Fill values for the current row.
                        for (int i = 0; i < fillers.Length; i++)
                        {
                            fillers[i].Set();
                        }
                    }
                }
            }
        }

        private static unsafe void SendViewToNativeAsCsr(IChannel ch, EnvironmentBlock* penv, IDataView view)
        {
            Contracts.AssertValue(ch);
            Contracts.Assert(penv != null);
            Contracts.AssertValue(view);
            if (penv->dataSink == null)
            {
                // Environment doesn't want any data!
                return;
            }

            var dataSink = MarshalDelegate<DataSink>(penv->dataSink);

            var schema = view.Schema;
            var colIndices = new List<int>();
            var outputDataKind = InternalDataKind.R4;

            int numOutputRows = 0;
            int numOutputCols = 0;

            for (int col = 0; col < schema.Count; col++)
            {
                if (schema[col].IsHidden)
                    continue;

                var fullType = schema[col].Type;
                var itemType = fullType.GetItemType();
                int valueCount = fullType.GetValueCount();

                if (valueCount == 0)
                {
                    throw ch.ExceptNotSupp("Column has variable length vector: " +
                        schema[col].Name + ". Not supported in python. Drop column before sending to Python");
                }

                if (itemType.IsStandardScalar())
                {
                    switch (itemType.GetRawKind())
                    {
                    default:
                        throw Contracts.Except("Data type {0} not supported", itemType.GetRawKind());

                    case InternalDataKind.I1:
                    case InternalDataKind.I2:
                    case InternalDataKind.U1:
                    case InternalDataKind.U2:
                    case InternalDataKind.R4:
                        break;

                    case InternalDataKind.I4:
                    case InternalDataKind.U4:
                    case InternalDataKind.I8:
                    case InternalDataKind.R8:
                        outputDataKind = InternalDataKind.R8;
                        break;
                    }
                }
                else
                {
                    throw Contracts.Except("Data type {0} not supported", itemType.GetRawKind());
                }

                colIndices.Add(col);
                numOutputCols += valueCount;
            }

            var nameIndices = new ValueListBuilder<int>(10);
            var nameUtf8Bytes = new ValueListBuilder<byte>(100);

            AddUniqueName("data", ref nameIndices, ref nameUtf8Bytes);
            AddUniqueName("indices", ref nameIndices, ref nameUtf8Bytes);
            AddUniqueName("indptr", ref nameIndices, ref nameUtf8Bytes);
            AddUniqueName("shape", ref nameIndices, ref nameUtf8Bytes);

            var kindList = new List<InternalDataKind> {outputDataKind,
                                                       InternalDataKind.I4,
                                                       InternalDataKind.I4,
                                                       InternalDataKind.I4};

            var valueCounts = new List<byte> { 1, 1, 1, 1 };

            var kinds = kindList.ToArray();
            var nameBytes = nameUtf8Bytes.AsSpan();
            var names = new byte*[nameIndices.Length];
            var valueCountsBytes = valueCounts.ToArray();

            fixed (InternalDataKind* prgkind = kinds)
            fixed (byte* prgbNames = nameBytes)
            fixed (byte** prgname = names)
            fixed (byte* prgbValueCount = valueCountsBytes)
            {
                for (int iid = 0; iid < names.Length; iid++)
                    names[iid] = prgbNames + nameIndices[iid];

                DataViewBlock block;
                block.ccol = nameIndices.Length;
                block.crow = view.GetRowCount() ?? 0;
                block.names = (sbyte**)prgname;
                block.kinds = prgkind;
                block.keyCards = null;
                block.valueCounts = prgbValueCount;

                dataSink(penv, &block, out var setters, out var keyValueSetter);

                if (setters == null) return;

                using (var cursor = view.GetRowCursor(view.Schema.Where(col => colIndices.Contains(col.Index))))
                {
                    CsrData csrData = new CsrData(penv, setters, outputDataKind);
                    var fillers = new CsrFillerBase[colIndices.Count];

                    for (int i = 0; i < colIndices.Count; i++)
                    {
                        var type = schema[colIndices[i]].Type;
                        fillers[i] = CsrFillerBase.Create(penv, cursor, colIndices[i], type, outputDataKind, csrData);
                    }

                    for (;; numOutputRows++)
                    {
                        if (!cursor.MoveNext()) break;

                        for (int i = 0; i < fillers.Length; i++)
                        {
                            fillers[i].Set();
                        }

                        csrData.IncrementRow();
                    }

                    csrData.SetShape(numOutputRows, numOutputCols);
                }
            }
        }

        private static void AddUniqueName(string name,
            ref ValueListBuilder<int> nameIndices, 
            ref ValueListBuilder<byte> utf8Names)
        {
            if (utf8Names.Capacity - utf8Names.Length < name.Length * 2 + 2)
                utf8Names.Grow();

            nameIndices.Append(utf8Names.Length);
            var bytesNumber = Encoding.UTF8.GetBytes(name, 0, name.Length, utf8Names.Buffer, utf8Names.Length);
            utf8Names.Length += bytesNumber;
            utf8Names.Append(0);
        }

        private static void AddUniqueName(
            string columnName,
            VBuffer<ReadOnlyMemory<char>> slotNames,
            ref ValueListBuilder<int> nameIndices,
            ref ValueListBuilder<byte> utf8Names)
        {
            var columnNameBytes = Encoding.UTF8.GetBytes(columnName);
            var dotBytes = Encoding.UTF8.GetBytes(".");

            foreach (var kvp in slotNames.Items(true))
            {
                // REVIEW: Add the proper number of zeros to the slot index to make them sort in the right order.
                var slotName = (!kvp.Value.IsEmpty ? kvp.Value.ToString() : kvp.Key.ToString(CultureInfo.InvariantCulture));
                if (utf8Names.Capacity - utf8Names.Length < slotName.Length * 2 + columnNameBytes.Length + dotBytes.Length)
                    utf8Names.Grow();
                nameIndices.Append(utf8Names.Length);
                utf8Names.AppendRange(columnNameBytes);
                utf8Names.AppendRange(dotBytes);
                var bytesNumber = Encoding.UTF8.GetBytes(slotName, 0, slotName.Length, utf8Names.Buffer, utf8Names.Length);
                utf8Names.Length += bytesNumber;
                utf8Names.Append(0);
            }
        }

        private abstract unsafe class BufferFillerBase
        {
            public delegate void ValuePoker<T>(T value, int col, long m, long n);

            protected readonly int _colIndex;
            protected readonly DataViewRow _input;

            protected BufferFillerBase(DataViewRow input, int pyColIndex)
            {
                _colIndex = pyColIndex;
                _input = input;
            }

            public static BufferFillerBase Create(EnvironmentBlock* penv, DataViewRow input, int pyCol, int idvCol, InternalDataKind dataKind, DataViewType type, void* setter)
            {
                var itemType = type.GetItemType();
                // We convert the unsigned types to signed types, with -1 indicating missing in Python.
                if (itemType.GetKeyCount() > 0)
                {
                    var keyCount = itemType.GetKeyCount();
                    uint keyMax = (uint)keyCount;
                    switch (itemType.GetRawKind())
                    {
                    case InternalDataKind.U1:
                        var fnI1 = MarshalDelegate<I1Setter>(setter);
                        ValuePoker<byte> pokeU1 =
                            (byte value, int col, long m, long n) => fnI1(penv, col, m, n, value > keyMax ? (sbyte)-1 : (sbyte)(value - 1));
                        return new Impl<byte>(input, pyCol, idvCol, type, pokeU1);
                    case InternalDataKind.U2:
                        var fnI2 = MarshalDelegate<I2Setter>(setter);
                        ValuePoker<ushort> pokeU2 =
                            (ushort value, int col, long m, long n) => fnI2(penv, col, m, n, value > keyMax ? (short)-1 : (short)(value - 1));
                        return new Impl<ushort>(input, pyCol, idvCol, type, pokeU2);
                    case InternalDataKind.U4:
                        var fnI4 = MarshalDelegate<I4Setter>(setter);
                        ValuePoker<uint> pokeU4 =
                            (uint value, int col, long m, long n) => fnI4(penv, col, m, n, value > keyMax ? -1 : (int)(value - 1));
                        return new Impl<uint>(input, pyCol, idvCol, type, pokeU4);
                    case InternalDataKind.U8:
                        // We convert U8 key types with key names to I4.
                        fnI4 = MarshalDelegate<I4Setter>(setter);
                        ValuePoker<ulong> pokeU8 =
                            (ulong value, int col, long m, long n) => fnI4(penv, col, m, n, value > keyMax ? -1 : (int)(value - 1));
                        return new Impl<ulong>(input, pyCol, idvCol, type, pokeU8);
                    }
                }
                // Key type with count=0
                else if (itemType is KeyDataViewType)
                {
                    switch (itemType.GetRawKind())
                    {
                    case InternalDataKind.U1:
                        var fnI1 = MarshalDelegate<I1Setter>(setter);
                        ValuePoker<byte> pokeU1 =
                            (byte value, int col, long m, long n) => fnI1(penv, col, m, n, (sbyte)(value - 1));
                        return new Impl<byte>(input, pyCol, idvCol, type, pokeU1);
                    case InternalDataKind.U2:
                        var fnI2 = MarshalDelegate<I2Setter>(setter);
                        ValuePoker<ushort> pokeU2 =
                            (ushort value, int col, long m, long n) => fnI2(penv, col, m, n, (short)(value - 1));
                        return new Impl<ushort>(input, pyCol, idvCol, type, pokeU2);
                    case InternalDataKind.U4:
                        var fnI4 = MarshalDelegate<I4Setter>(setter);
                        ValuePoker<uint> pokeU4 =
                            (uint value, int col, long m, long n) => fnI4(penv, col, m, n, (int)(value - 1));
                        return new Impl<uint>(input, pyCol, idvCol, type, pokeU4);
                    case InternalDataKind.U8:
                        // We convert U8 key types with key names to I4.
                        fnI4 = MarshalDelegate<I4Setter>(setter);
                        ValuePoker<ulong> pokeU8 =
                            (ulong value, int col, long m, long n) => fnI4(penv, col, m, n, (int)(value - 1));
                        return new Impl<ulong>(input, pyCol, idvCol, type, pokeU8);
                    }
                }
                else
                {
                    switch (dataKind)
                    {
                    case InternalDataKind.R4:
                        var fnR4 = MarshalDelegate<R4Setter>(setter);
                        ValuePoker<float> pokeR4 =
                            (float value, int col, long m, long n) => fnR4(penv, col, m, n, value);
                        return new Impl<float>(input, pyCol, idvCol, type, pokeR4);
                    case InternalDataKind.R8:
                        var fnR8 = MarshalDelegate<R8Setter>(setter);
                        ValuePoker<double> pokeR8 =
                            (double value, int col, long m, long n) => fnR8(penv, col, m, n, value);
                        return new Impl<double>(input, pyCol, idvCol, type, pokeR8);
                    case InternalDataKind.BL:
                        var fnBl = MarshalDelegate<BLSetter>(setter);
                        ValuePoker<bool> pokeBl =
                            (bool value, int col, long m, long n) => fnBl(penv, col, m, n, !value ? (byte)0 : value ? (byte)1 : (byte)0xFF);
                        return new Impl<bool>(input, pyCol, idvCol, type, pokeBl);
                    case InternalDataKind.I1:
                        var fnI1 = MarshalDelegate<I1Setter>(setter);
                        ValuePoker<sbyte> pokeI1 =
                            (sbyte value, int col, long m, long n) => fnI1(penv, col, m, n, value);
                        return new Impl<sbyte>(input, pyCol, idvCol, type, pokeI1);
                    case InternalDataKind.I2:
                        var fnI2 = MarshalDelegate<I2Setter>(setter);
                        ValuePoker<short> pokeI2 =
                            (short value, int col, long m, long n) => fnI2(penv, col, m, n, value);
                        return new Impl<short>(input, pyCol, idvCol, type, pokeI2);
                    case InternalDataKind.I4:
                        var fnI4 = MarshalDelegate<I4Setter>(setter);
                        ValuePoker<int> pokeI4 =
                            (int value, int col, long m, long n) => fnI4(penv, col, m, n, value);
                        return new Impl<int>(input, pyCol, idvCol, type, pokeI4);
                    case InternalDataKind.I8:
                        var fnI8 = MarshalDelegate<I8Setter>(setter);
                        ValuePoker<long> pokeI8 =
                            (long value, int col, long m, long n) => fnI8(penv, col, m, n, value);
                        return new Impl<long>(input, pyCol, idvCol, type, pokeI8);
                    case InternalDataKind.U1:
                        var fnU1 = MarshalDelegate<U1Setter>(setter);
                        ValuePoker<byte> pokeU1 =
                            (byte value, int col, long m, long n) => fnU1(penv, col, m, n, value);
                        return new Impl<byte>(input, pyCol, idvCol, type, pokeU1);
                    case InternalDataKind.U2:
                        var fnU2 = MarshalDelegate<U2Setter>(setter);
                        ValuePoker<ushort> pokeU2 =
                            (ushort value, int col, long m, long n) => fnU2(penv, col, m, n, value);
                        return new Impl<ushort>(input, pyCol, idvCol, type, pokeU2);
                    case InternalDataKind.U4:
                        var fnU4 = MarshalDelegate<U4Setter>(setter);
                        ValuePoker<uint> pokeU4 =
                            (uint value, int col, long m, long n) => fnU4(penv, col, m, n, value);
                        return new Impl<uint>(input, pyCol, idvCol, type, pokeU4);
                    case InternalDataKind.U8:
                        var fnU8 = MarshalDelegate<U8Setter>(setter);
                        ValuePoker<ulong> pokeU8 =
                            (ulong value, int col, long m, long n) => fnU8(penv, col, m, n, value);
                        return new Impl<ulong>(input, pyCol, idvCol, type, pokeU8);
                    case InternalDataKind.DT:
                        var fnDT = MarshalDelegate<I8Setter>(setter);
                        ValuePoker<DateTime> pokeDT =
                            (DateTime value, int col, long m, long n) =>
                            {
                                DateTimeOffset dto = (value.Kind == DateTimeKind.Unspecified) ? 
                                                     new DateTimeOffset(value, TimeSpan.Zero) :
                                                     new DateTimeOffset(value);
                                fnDT(penv, col, m, n, dto.ToUnixTimeMilliseconds());
                            };
                        return new Impl<DateTime>(input, pyCol, idvCol, type, pokeDT);
                    case InternalDataKind.TX:
                        var fnTX = MarshalDelegate<TXSetter>(setter);
                        ValuePoker<ReadOnlyMemory<char>> pokeTX =
                            (ReadOnlyMemory<char> value, int col, long m, long n) =>
                            {
                                if (value.IsEmpty)
                                    fnTX(penv, col, m, n, null, 0);
                                else
                                {
                                    byte[] bt = Encoding.UTF8.GetBytes(value.ToString());
                                    fixed (byte* pt = bt)
                                        fnTX(penv, col, m, n, (sbyte*)pt, bt.Length);
                                }
                            };
                        return new Impl<ReadOnlyMemory<char>>(input, pyCol, idvCol, type, pokeTX);
                    default:
                        throw Contracts.Except("Data type not handled");
                    }
                }

                Contracts.Assert(false, "Unhandled type!");
                return null;
            }

            public abstract void Set();

            private sealed class Impl<TSrc> : BufferFillerBase
            {
                private readonly ValueGetter<VBuffer<TSrc>> _getVec;
                private VBuffer<TSrc> _buffer;
                private readonly ValueGetter<TSrc> _get;
                private readonly ValuePoker<TSrc> _poker;
                private readonly bool _isVarLength;

                public Impl(DataViewRow input, int pyColIndex, int idvColIndex, DataViewType type, ValuePoker<TSrc> poker)
                    : base(input, pyColIndex)
                {
                    Contracts.AssertValue(input);
                    Contracts.Assert(0 <= idvColIndex && idvColIndex < input.Schema.Count);

                    if (type is VectorDataViewType)
                        _getVec = RowCursorUtils.GetVecGetterAs<TSrc>((PrimitiveDataViewType)type.GetItemType(), input, idvColIndex);
                    else
                        _get = RowCursorUtils.GetGetterAs<TSrc>(type, input, idvColIndex);

                    _poker = poker;
                    _isVarLength = (type.GetValueCount() == 0);
                }
                public override void Set()
                {
                    if (_getVec != null)
                    {
                        _getVec(ref _buffer);
                        if (_buffer.IsDense)
                        {
                            for (int i = 0; i < _buffer.Length; i++)
                            {
                                if (_isVarLength)
                                     _poker(_buffer.GetValues()[i], _colIndex, _input.Position, i);
                                else _poker(_buffer.GetValues()[i], _colIndex + i, _input.Position, 0);
                            }
                        }
                        else
                        {
                            int ii = 0;
                            var values = _buffer.GetValues();
                            var indices = _buffer.GetIndices();
                            for (int i = 0; i < _buffer.Length; i++)
                            {
                                while (ii < values.Length && indices[ii] < i)
                                    ii++;
                                TSrc val = default(TSrc);
                                if (ii < values.Length && indices[ii] == i)
                                    val = values[ii];

                                if (_isVarLength)
                                     _poker(val, _colIndex, _input.Position, i);
                                else _poker(val, _colIndex + i, _input.Position, 0);
                            }
                        }
                    }
                    else
                    {
                        TSrc value = default(TSrc);
                        _get(ref value);
                        _poker(value, _colIndex, _input.Position, 0);
                    }
                }
            }
        }

        private unsafe class CsrData
        {
            private const int DataCol = 0;
            private const int IndicesCol = 1;
            private const int IndPtrCol = 2;
            private const int ShapeCol = 3;

            private readonly R4Setter _r4DataSetter;
            private readonly R8Setter _r8DataSetter;
            private readonly I4Setter _indicesSetter;
            private readonly I4Setter _indptrSetter;
            private readonly I4Setter _shapeSetter;

            public int col;

            private int _row;
            private int _index;

            private EnvironmentBlock* _penv;

            public CsrData(EnvironmentBlock* penv, void** setters, InternalDataKind outputDataKind)
            {
                col = 0;

                _row = 0;
                _index = 0;
                _penv = penv;

                if (outputDataKind == InternalDataKind.R4)
                {
                    _r4DataSetter = MarshalDelegate<R4Setter>(setters[DataCol]);
                    _r8DataSetter = null;
                }
                else if(outputDataKind == InternalDataKind.R8)
                {
                    _r4DataSetter = null;
                    _r8DataSetter = MarshalDelegate<R8Setter>(setters[DataCol]);
                }

                _indicesSetter = MarshalDelegate<I4Setter>(setters[IndicesCol]);
                _indptrSetter = MarshalDelegate<I4Setter>(setters[IndPtrCol]);
                _shapeSetter = MarshalDelegate<I4Setter>(setters[ShapeCol]);

                _indptrSetter(_penv, IndPtrCol, 0, 0, 0);
            }

            public void AppendR4(float value, int col)
            {
                _r4DataSetter(_penv, DataCol, _index, 0, value);
                _indicesSetter(_penv, IndicesCol, _index, 0, col);
                _index++;
            }

            public void AppendR8(double value, int col)
            {
                _r8DataSetter(_penv, DataCol, _index, 0, value);
                _indicesSetter(_penv, IndicesCol, _index, 0, col);
                _index++;
            }

            public void IncrementRow()
            {
                col = 0;
                _row++;

                _indptrSetter(_penv, IndPtrCol, _row, 0, _index);
            }

            public void SetShape(int m, int n)
            {
                _shapeSetter(_penv, ShapeCol, 0, 0, m);
                _shapeSetter(_penv, ShapeCol, 1, 0, n);
            }
        }

        private abstract unsafe class CsrFillerBase
        {
            public delegate void DataAppender<T>(T value, int col);

            protected CsrFillerBase() {}

            public static CsrFillerBase Create(EnvironmentBlock* penv,
                                               DataViewRow input,
                                               int idvCol,
                                               DataViewType idvColType,
                                               InternalDataKind outputDataKind,
                                               CsrData csrData)
            {
                if (outputDataKind == InternalDataKind.R4)
                {
                    switch (idvColType.GetItemType().GetRawKind())
                    {
                    case InternalDataKind.I1:
                        DataAppender<sbyte> appendI1 = (sbyte val, int i) => csrData.AppendR4((float)val, i);
                        return new CsrFiller<sbyte>(input, idvCol, idvColType, appendI1, csrData);
                    case InternalDataKind.I2:
                        DataAppender<short> appendI2 = (short val, int i) => csrData.AppendR4((float)val, i);
                        return new CsrFiller<short>(input, idvCol, idvColType, appendI2, csrData);
                    case InternalDataKind.U1:
                        DataAppender<byte> appendU1 = (byte val, int i) => csrData.AppendR4((float)val, i);
                        return new CsrFiller<byte>(input, idvCol, idvColType, appendU1, csrData);
                    case InternalDataKind.U2:
                        DataAppender<ushort> appendU2 = (ushort val, int i) => csrData.AppendR4((float)val, i);
                        return new CsrFiller<ushort>(input, idvCol, idvColType, appendU2, csrData);
                    case InternalDataKind.R4:
                        DataAppender<float> appendR4 = (float val, int i) => csrData.AppendR4((float)val, i);
                        return new CsrFiller<float>(input, idvCol, idvColType, appendR4, csrData);
                    default:
                        throw Contracts.Except("Source data type not supported");
                    }
                }
                else if (outputDataKind == InternalDataKind.R8)
                {
                    switch (idvColType.GetItemType().GetRawKind())
                    {
                    case InternalDataKind.I1:
                        DataAppender<sbyte> appendI1 = (sbyte val, int i) => csrData.AppendR8((double)val, i);
                        return new CsrFiller<sbyte>(input, idvCol, idvColType, appendI1, csrData);
                    case InternalDataKind.I2:
                        DataAppender<short> appendI2 = (short val, int i) => csrData.AppendR8((double)val, i);
                        return new CsrFiller<short>(input, idvCol, idvColType, appendI2, csrData);
                    case InternalDataKind.I4:
                        DataAppender<int> appendI4 = (int val, int i) => csrData.AppendR8((double)val, i);
                        return new CsrFiller<int>(input, idvCol, idvColType, appendI4, csrData);
                    case InternalDataKind.U1:
                        DataAppender<byte> appendU1 = (byte val, int i) => csrData.AppendR8((double)val, i);
                        return new CsrFiller<byte>(input, idvCol, idvColType, appendU1, csrData);
                    case InternalDataKind.U2:
                        DataAppender<ushort> appendU2 = (ushort val, int i) => csrData.AppendR8((double)val, i);
                        return new CsrFiller<ushort>(input, idvCol, idvColType, appendU2, csrData);
                    case InternalDataKind.U4:
                        DataAppender<uint> appendU4 = (uint val, int i) => csrData.AppendR8((double)val, i);
                        return new CsrFiller<uint>(input, idvCol, idvColType, appendU4, csrData);
                    case InternalDataKind.R4:
                        DataAppender<float> appendR4 = (float val, int i) => csrData.AppendR8((double)val, i);
                        return new CsrFiller<float>(input, idvCol, idvColType, appendR4, csrData);
                    case InternalDataKind.I8:
                        DataAppender<long> appendI8 = (long val, int i) => csrData.AppendR8((double)val, i);
                        return new CsrFiller<long>(input, idvCol, idvColType, appendI8, csrData);
                    case InternalDataKind.R8:
                        DataAppender<double> appendR8 = (double val, int i) => csrData.AppendR8((double)val, i);
                        return new CsrFiller<double>(input, idvCol, idvColType, appendR8, csrData);
                    default:
                        throw Contracts.Except("Source data type not supported");
                    }
                }

                throw Contracts.Except("Target data type not supported.");
            }

            public abstract void Set();

            private sealed class CsrFiller<TSrc> : CsrFillerBase
            {
                private readonly ValueGetter<VBuffer<TSrc>> _getVec;
                private readonly ValueGetter<TSrc> _get;
                private VBuffer<TSrc> _buffer;

                private CsrData _csrData;
                private readonly DataAppender<TSrc> _dataAppender;

                private readonly IEqualityComparer<TSrc> comparer = EqualityComparer<TSrc>.Default;

                public CsrFiller(DataViewRow input,
                                 int idvColIndex,
                                 DataViewType type,
                                 DataAppender<TSrc> dataAppender,
                                 CsrData csrData)
                    : base()
                {
                    Contracts.AssertValue(input);
                    Contracts.Assert(0 <= idvColIndex && idvColIndex < input.Schema.Count);

                    if (type is VectorDataViewType)
                        _getVec = RowCursorUtils.GetVecGetterAs<TSrc>((PrimitiveDataViewType)type.GetItemType(), input, idvColIndex);
                    else
                        _get = RowCursorUtils.GetGetterAs<TSrc>(type, input, idvColIndex);

                    _csrData = csrData;
                    _dataAppender = dataAppender;
                }

                public bool IsDefault(TSrc t)
                {
                    return comparer.Equals(t, default(TSrc));
                }

                public override void Set()
                {
                    if (_getVec != null)
                    {
                        _getVec(ref _buffer);
                        if (_buffer.IsDense)
                        {
                            var values = _buffer.GetValues();

                            for (int i = 0; i < values.Length; i++)
                            {
                                if (!IsDefault(values[i]))
                                    _dataAppender(values[i], _csrData.col);

                                _csrData.col++;
                            }
                        }
                        else
                        {
                            var values = _buffer.GetValues();
                            var indices = _buffer.GetIndices();

                            for (int i = 0; i < values.Length; i++)
                            {
                                if (!IsDefault(values[i]))
                                    _dataAppender(values[i], _csrData.col + indices[i]);
                            }

                            _csrData.col += _buffer.Length;
                        }
                    }
                    else
                    {
                        TSrc value = default(TSrc);
                        _get(ref value);

                        if (!IsDefault(value))
                            _dataAppender(value, _csrData.col);

                        _csrData.col++;
                    }
                }
            }
        }
    }
}
