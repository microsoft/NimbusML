//------------------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//------------------------------------------------------------------------------

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Microsoft.MachineLearning.DotNetBridge
{
    public unsafe static partial class Bridge
    {
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
            public readonly long* ids;
            [FieldOffset(0x18)]
            public readonly sbyte** names;
            [FieldOffset(0x20)]
            public readonly DataKind* kinds;
            [FieldOffset(0x28)]
            public readonly long* keyCards;
            [FieldOffset(0x30)]
            public readonly long* vecCards;
            [FieldOffset(0x38)]
            public readonly void** getters;

            // Call back pointers.
            [FieldOffset(0x40)]
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
            public DataKind* kinds;

            // For columns that have key type, these contain the cardinalities of the
            // key types. Zero means unbounded, -1 means not a key type.
            [FieldOffset(0x20)]
            public int* keyCards;
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

        private static unsafe void SendViewToNative(IChannel ch, EnvironmentBlock* penv, IDataView view, Dictionary<string, ColumnMetadataInfo> infos = null)
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
            var colIndices = new List<int>();
            var kindList = new List<DataKind>();
            var keyCardList = new List<int>();
            var nameUtf8Bytes = new List<Byte>();
            var nameIndices = new List<int>();

            var expandCols = new HashSet<int>();
            var allNames = new HashSet<string>();

            for (int col = 0; col < schema.ColumnCount; col++)
            {
                if (schema.IsHidden(col))
                    continue;

                var fullType = schema.GetColumnType(col);
                var itemType = fullType.ItemType;
                var name = schema.GetColumnName(col);

                DataKind kind = itemType.RawKind;
                int keyCard;

                if (fullType.ValueCount == 0)
                {
                    throw ch.ExceptNotSupp("Column has variable length vector: " + 
                        name + ". Not supported in python. Drop column before sending to Python");
                }

                if (itemType.IsKey)
                {
                    // Key types are returned as their signed counterparts in Python, so that -1 can be the missing value.
                    // For U1 and U2 kinds, we convert to a larger type to prevent overflow. For U4 and U8 kinds, we convert
                    // to I4 if the key count is known (since KeyCount is an I4), and to I8 otherwise.
                    switch (kind)
                    {
                    case DataKind.U1:
                        kind = DataKind.I2;
                        break;
                    case DataKind.U2:
                        kind = DataKind.I4;
                        break;
                    case DataKind.U4:
                        // We convert known-cardinality U4 key types to I4.
                        kind = itemType.KeyCount > 0 ? DataKind.I4 : DataKind.I8;
                        break;
                    case DataKind.U8:
                        // We convert known-cardinality U8 key types to I4.
                        kind = itemType.KeyCount > 0 ? DataKind.I4 : DataKind.I8;
                        break;
                    }

                    keyCard = itemType.KeyCount;
                    if (!schema.HasKeyNames(col, keyCard))
                        keyCard = -1;
                }
                else if (itemType.IsStandardScalar)
                {
                    switch (itemType.RawKind)
                    {
                    default:
                        throw Contracts.Except("Data type {0} not handled", itemType.RawKind);

                    case DataKind.I1:
                    case DataKind.I2:
                    case DataKind.I4:
                    case DataKind.I8:
                    case DataKind.U1:
                    case DataKind.U2:
                    case DataKind.U4:
                    case DataKind.U8:
                    case DataKind.R4:
                    case DataKind.R8:
                    case DataKind.BL:
                    case DataKind.TX:
                        break;
                    }
                    keyCard = -1;
                }
                else
                {
                    throw Contracts.Except("Data type {0} not handled", itemType.RawKind);
                }

                int nSlots;
                ColumnMetadataInfo info;
                if (infos != null && infos.TryGetValue(name, out info) && info.Expand)
                {
                    expandCols.Add(col);
                    Contracts.Assert(fullType.IsKnownSizeVector);
                    nSlots = fullType.VectorSize;
                    if (info.SlotNames != null)
                    {
                        Contracts.Assert(info.SlotNames.Length == nSlots);
                        for (int i = 0; i < nSlots; i++)
                            AddUniqueName(info.SlotNames[i], allNames, nameIndices, nameUtf8Bytes);
                    }
                    else if (schema.HasSlotNames(col, nSlots))
                    {
                        var romNames = default(VBuffer<ReadOnlyMemory<char>>);
                        schema.GetMetadata(MetadataUtils.Kinds.SlotNames, col, ref romNames);
                        foreach (var kvp in romNames.Items(true))
                        {
                            // REVIEW: Add the proper number of zeros to the slot index to make them sort in the right order.
                            var slotName = name + "." +
                                (!kvp.Value.IsEmpty ? kvp.Value.ToString() : kvp.Key.ToString(CultureInfo.InvariantCulture));
                            AddUniqueName(slotName, allNames, nameIndices, nameUtf8Bytes);
                        }
                    }
                    else
                    {
                        for (int i = 0; i < nSlots; i++)
                            AddUniqueName(name + "." + i, allNames, nameIndices, nameUtf8Bytes);
                    }
                }
                else
                {
                    nSlots = 1;
                    AddUniqueName(name, allNames, nameIndices, nameUtf8Bytes);
                }

                colIndices.Add(col);
                for (int i = 0; i < nSlots; i++)
                {
                    kindList.Add(kind);
                    keyCardList.Add(keyCard);
                }
            }

            ch.Assert(allNames.Count == kindList.Count);
            ch.Assert(allNames.Count == keyCardList.Count);
            ch.Assert(allNames.Count == nameIndices.Count);

            var kinds = kindList.ToArray();
            var keyCards = keyCardList.ToArray();
            var nameBytes = nameUtf8Bytes.ToArray();
            var names = new byte*[allNames.Count];

            fixed (DataKind* prgkind = kinds)
            fixed (byte* prgbNames = nameBytes)
            fixed (byte** prgname = names)
            fixed (int* prgkeyCard = keyCards)
            {
                for (int iid = 0; iid < names.Length; iid++)
                    names[iid] = prgbNames + nameIndices[iid];

                DataViewBlock block;
                block.ccol = allNames.Count;
                block.crow = view.GetRowCount() ?? 0;
                block.names = (sbyte**)prgname;
                block.kinds = prgkind;
                block.keyCards = prgkeyCard;

                dataSink(penv, &block, out var setters, out var keyValueSetter);

                if (setters == null)
                {
                    // REVIEW: What should we do?
                    return;
                }
                ch.Assert(keyValueSetter != null);
                var kvSet = MarshalDelegate<KeyValueSetter>(keyValueSetter);
                using (var cursor = view.GetRowCursor(colIndices.Contains))
                {
                    var fillers = new BufferFillerBase[colIndices.Count];
                    var pyColumn = 0;
                    var keyIndex = 0;
                    for (int i = 0; i < colIndices.Count; i++)
                    {
                        var type = schema.GetColumnType(colIndices[i]);
                        if (type.ItemType.IsKey && schema.HasKeyNames(colIndices[i], type.ItemType.KeyCount))
                        {
                            ch.Assert(schema.HasKeyNames(colIndices[i], type.ItemType.KeyCount));
                            var keyValues = default(VBuffer<ReadOnlyMemory<char>>);
                            schema.GetMetadata(MetadataUtils.Kinds.KeyValues, colIndices[i], ref keyValues);
                            for (int slot = 0; slot < type.ValueCount; slot++)
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
                        fillers[i] = BufferFillerBase.Create(penv, cursor, pyColumn, colIndices[i], kinds[pyColumn], type, setters[pyColumn]);
                        pyColumn += type.IsVector ? type.VectorSize : 1;
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

        private static string AddUniqueName(string name, HashSet<string> allNames, List<int> nameIndices, List<Byte> nameUtf8Bytes)
        {
            string newName = name;
            int i = 1;
            while (!allNames.Add(newName))
                newName = string.Format(CultureInfo.InvariantCulture, "{0}_{1}", name, i++);
            // REVIEW: Column names should not be affected by the slot names. They should always win against slot names.
            byte[] bNewName = Encoding.UTF8.GetBytes(newName);
            nameIndices.Add(nameUtf8Bytes.Count);
            nameUtf8Bytes.AddRange(bNewName);
            nameUtf8Bytes.Add(0);
            return newName;
        }

        private abstract unsafe class BufferFillerBase
        {
            public delegate void ValuePoker<T>(T value, int col, long index);

            protected readonly int _colIndex;
            protected readonly IRow _input;

            protected BufferFillerBase(IRow input, int pyColIndex)
            {
                _colIndex = pyColIndex;
                _input = input;
            }

            public static BufferFillerBase Create(EnvironmentBlock* penv, IRow input, int pyCol, int idvCol, DataKind dataKind, ColumnType type, void* setter)
            {
                var itemType = type.ItemType;
                // We convert the unsigned types to signed types, with -1 indicating missing in Python.
                if (itemType.KeyCount > 0)
                {
                    var keyCount = itemType.KeyCount;
                    uint keyMax = (uint)keyCount;
                    switch (itemType.RawKind)
                    {
                    case DataKind.U1:
                        var fnI1 = MarshalDelegate<I1Setter>(setter);
                        ValuePoker<byte> pokeU1 =
                            (byte value, int col, long index) => fnI1(penv, col, index, value > keyMax ? (sbyte)-1 : (sbyte)(value - 1));
                        return new Impl<byte>(input, pyCol, idvCol, type, pokeU1);
                    case DataKind.U2:
                        var fnI2 = MarshalDelegate<I2Setter>(setter);
                        ValuePoker<ushort> pokeU2 =
                            (ushort value, int col, long index) => fnI2(penv, col, index, value > keyMax ? (short)-1 : (short)(value - 1));
                        return new Impl<ushort>(input, pyCol, idvCol, type, pokeU2);
                    case DataKind.U4:
                        var fnI4 = MarshalDelegate<I4Setter>(setter);
                        ValuePoker<uint> pokeU4 =
                            (uint value, int col, long index) => fnI4(penv, col, index, value > keyMax ? -1 : (int)(value - 1));
                        return new Impl<uint>(input, pyCol, idvCol, type, pokeU4);
                    case DataKind.U8:
                        // We convert U8 key types with key names to I4.
                        fnI4 = MarshalDelegate<I4Setter>(setter);
                        ValuePoker<ulong> pokeU8 =
                            (ulong value, int col, long index) => fnI4(penv, col, index, value > keyMax ? -1 : (int)(value - 1));
                        return new Impl<ulong>(input, pyCol, idvCol, type, pokeU8);
                    }
                }
                // Key type with count=0
                else if (itemType.IsKey)
                {
                    switch (itemType.RawKind)
                    {
                    case DataKind.U1:
                        var fnI1 = MarshalDelegate<I1Setter>(setter);
                        ValuePoker<byte> pokeU1 =
                            (byte value, int col, long index) => fnI1(penv, col, index, (sbyte)(value - 1));
                        return new Impl<byte>(input, pyCol, idvCol, type, pokeU1);
                    case DataKind.U2:
                        var fnI2 = MarshalDelegate<I2Setter>(setter);
                        ValuePoker<ushort> pokeU2 =
                            (ushort value, int col, long index) => fnI2(penv, col, index, (short)(value - 1));
                        return new Impl<ushort>(input, pyCol, idvCol, type, pokeU2);
                    case DataKind.U4:
                        var fnI4 = MarshalDelegate<I4Setter>(setter);
                        ValuePoker<uint> pokeU4 =
                            (uint value, int col, long index) => fnI4(penv, col, index, (int)(value - 1));
                        return new Impl<uint>(input, pyCol, idvCol, type, pokeU4);
                    case DataKind.U8:
                        // We convert U8 key types with key names to I4.
                        fnI4 = MarshalDelegate<I4Setter>(setter);
                        ValuePoker<ulong> pokeU8 =
                            (ulong value, int col, long index) => fnI4(penv, col, index, (int)(value - 1));
                        return new Impl<ulong>(input, pyCol, idvCol, type, pokeU8);
                    }
                }
                else
                {
                    switch (dataKind)
                    {
                    case DataKind.R4:
                        var fnR4 = MarshalDelegate<R4Setter>(setter);
                        ValuePoker<float> pokeR4 =
                            (float value, int col, long index) => fnR4(penv, col, index, value);
                        return new Impl<float>(input, pyCol, idvCol, type, pokeR4);
                    case DataKind.R8:
                        var fnR8 = MarshalDelegate<R8Setter>(setter);
                        ValuePoker<double> pokeR8 =
                            (double value, int col, long index) => fnR8(penv, col, index, value);
                        return new Impl<double>(input, pyCol, idvCol, type, pokeR8);
                    case DataKind.BL:
                        var fnBl = MarshalDelegate<BLSetter>(setter);
                        ValuePoker<bool> pokeBl =
                            (bool value, int col, long index) => fnBl(penv, col, index, !value ? (byte)0 : value ? (byte)1 : (byte)0xFF);
                        return new Impl<bool>(input, pyCol, idvCol, type, pokeBl);
                    case DataKind.I1:
                        var fnI1 = MarshalDelegate<I1Setter>(setter);
                        ValuePoker<sbyte> pokeI1 =
                            (sbyte value, int col, long index) => fnI1(penv, col, index, value);
                        return new Impl<sbyte>(input, pyCol, idvCol, type, pokeI1);
                    case DataKind.I2:
                        var fnI2 = MarshalDelegate<I2Setter>(setter);
                        ValuePoker<short> pokeI2 =
                            (short value, int col, long index) => fnI2(penv, col, index, value);
                        return new Impl<short>(input, pyCol, idvCol, type, pokeI2);
                    case DataKind.I4:
                        var fnI4 = MarshalDelegate<I4Setter>(setter);
                        ValuePoker<int> pokeI4 =
                            (int value, int col, long index) => fnI4(penv, col, index, value);
                        return new Impl<int>(input, pyCol, idvCol, type, pokeI4);
                    case DataKind.I8:
                        var fnI8 = MarshalDelegate<I8Setter>(setter);
                        ValuePoker<long> pokeI8 =
                            (long value, int col, long index) => fnI8(penv, col, index, value);
                        return new Impl<long>(input, pyCol, idvCol, type, pokeI8);
                    case DataKind.U1:
                        var fnU1 = MarshalDelegate<U1Setter>(setter);
                        ValuePoker<byte> pokeU1 =
                            (byte value, int col, long index) => fnU1(penv, col, index, value);
                        return new Impl<byte>(input, pyCol, idvCol, type, pokeU1);
                    case DataKind.U2:
                        var fnU2 = MarshalDelegate<U2Setter>(setter);
                        ValuePoker<ushort> pokeU2 =
                            (ushort value, int col, long index) => fnU2(penv, col, index, value);
                        return new Impl<ushort>(input, pyCol, idvCol, type, pokeU2);
                    case DataKind.U4:
                        var fnU4 = MarshalDelegate<U4Setter>(setter);
                        ValuePoker<uint> pokeU4 =
                            (uint value, int col, long index) => fnU4(penv, col, index, value);
                        return new Impl<uint>(input, pyCol, idvCol, type, pokeU4);
                    case DataKind.U8:
                        var fnU8 = MarshalDelegate<U8Setter>(setter);
                        ValuePoker<ulong> pokeU8 =
                            (ulong value, int col, long index) => fnU8(penv, col, index, value);
                        return new Impl<ulong>(input, pyCol, idvCol, type, pokeU8);
                    case DataKind.TX:
                        var fnTX = MarshalDelegate<TXSetter>(setter);
                        ValuePoker<ReadOnlyMemory<char>> pokeTX =
                            (ReadOnlyMemory<char> value, int col, long index) =>
                            {
                                if (value.IsEmpty)
                                    fnTX(penv, col, index, null, 0);
                                else
                                {
                                    byte[] bt = Encoding.UTF8.GetBytes(value.ToString());
                                    fixed (byte* pt = bt)
                                        fnTX(penv, col, index, (sbyte*)pt, bt.Length);
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

                public Impl(IRow input, int pyColIndex, int idvColIndex, ColumnType type, ValuePoker<TSrc> poker)
                    : base(input, pyColIndex)
                {
                    Contracts.AssertValue(input);
                    Contracts.Assert(0 <= idvColIndex && idvColIndex < input.Schema.ColumnCount);

                    if (type.IsVector)
                        _getVec = RowCursorUtils.GetVecGetterAs<TSrc>(type.ItemType.AsPrimitive, input, idvColIndex);
                    else
                        _get = RowCursorUtils.GetGetterAs<TSrc>(type, input, idvColIndex);

                    _poker = poker;
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
                                _poker(_buffer.Values[i], _colIndex + i, _input.Position);
                            }
                        }
                        else
                        {
                            int ii = 0;
                            for (int i = 0; i < _buffer.Length; i++)
                            {
                                while (ii < _buffer.Count && _buffer.Indices[ii] < i)
                                    ii++;
                                TSrc val = default(TSrc);
                                if (ii < _buffer.Count && _buffer.Indices[ii] == i)
                                    val = _buffer.Values[ii];
                                _poker(val, _colIndex + i, _input.Position);
                            }
                        }
                    }
                    else
                    {
                        TSrc value = default(TSrc);
                        _get(ref value);
                        _poker(value, _colIndex, _input.Position);
                    }
                }
            }
        }
    }
}
