//------------------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//------------------------------------------------------------------------------

using System;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.Linq;
using System.Threading;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;

namespace Microsoft.MachineLearning.DotNetBridge
{
    public unsafe static partial class Bridge
    {
        private sealed class NativeDataView : IDataView, IDisposable
        {
            private const int BatchSize = 64;

            private readonly long _rowCount;
            private readonly Column[] _columns;

            private readonly IHost _host;

            public bool CanShuffle => false;

            /// This field contains some information copied from <see cref="_columns">.
            /// For example, <see cref="Schema">[i].Name is the same to <see cref="_columns">[i].DetachedColumn.Name.
            /// This is a by-product of using the new <see cref="ML.Data.Schema"/> API. As a compromise, 
            /// instead of changing all <see cref="Column"/> derived classes,
            /// we decided to keep this duplicate piece of data as a quick solution.
            public Schema Schema { get; }

            public NativeDataView(IHostEnvironment env, DataSourceBlock* pdata)
            {
                Contracts.AssertValue(env);
                _host = env.Register("PyNativeDataView");
                _rowCount = pdata->crow;

                var nameSet = new HashSet<string>();
                var columns = new List<Column>();
                for (int c = 0; c < pdata->ccol; c++)
                {
                    string name = Bridge.BytesToString(pdata->names[c]);
                    // Names must be non-null && non-empty unique.
                    Contracts.CheckParam(!string.IsNullOrWhiteSpace(name), "name");

                    // Names must be unique.
                    if (!nameSet.Add(name))
                        throw _host.ExceptUserArg("name", "Duplicate column name: {0}", name);

                    switch (pdata->kinds[c])
                    {
                        default:
                            _host.Assert(false);
                            break;
                        case DataKind.BL:
                            if (pdata->vecCards[c] == -1)
                                columns.Add(new BoolColumn(pdata, pdata->getters[c], c, name));
                            else
                                columns.Add(new VectorBoolColumn(pdata, pdata->getters[c], c, name, new VectorType(BoolType.Instance, (int)pdata->vecCards[c])));
                            break;
                        case DataKind.U1:
                            // catch if categoricals are passed by other than U4 types
                            Contracts.Assert(pdata->keyCards[c] <= 0);
                            if (pdata->vecCards[c] == -1)
                                columns.Add(new U1Column(pdata, pdata->getters[c], c, name));
                            else
                                columns.Add(new VectorUInt1Column(pdata, pdata->getters[c], c, name, new VectorType(NumberType.U1, (int)pdata->vecCards[c])));
                            break;
                        case DataKind.U2:
                            // catch if categoricals are passed by other than U4 types
                            Contracts.Assert(pdata->keyCards[c] <= 0);
                            if (pdata->vecCards[c] == -1)
                                columns.Add(new U2Column(pdata, pdata->getters[c], c, name));
                            else
                                columns.Add(new VectorUInt2Column(pdata, pdata->getters[c], c, name, new VectorType(NumberType.U2, (int)pdata->vecCards[c])));
                            break;
                        case DataKind.U4:
                            if (pdata->keyCards[c] > 0)
                            {
                                // Categoricals from python are passed as U4 type
                                var keyNames = default(VBuffer<ReadOnlyMemory<char>>);
                                int keyCount = pdata->keyCards[c] <= int.MaxValue ? (int)pdata->keyCards[c] : 0;
                                if (!GetLabels(pdata, c, keyCount, ref keyNames))
                                    env.Assert(keyNames.Length == 0);
                                columns.Add(new KeyColumn(pdata, pdata->getters[c], c, name, keyCount, ref keyNames));
                            }
                            else if (pdata->vecCards[c] == -1)
                                columns.Add(new U4Column(pdata, pdata->getters[c], c, name));
                            else
                                columns.Add(new VectorUInt4Column(pdata, pdata->getters[c], c, name, new VectorType(NumberType.U4, (int)pdata->vecCards[c])));
                            break;
                        case DataKind.U8:
                            // catch if categoricals are passed by other than U4 types
                            Contracts.Assert(pdata->keyCards[c] <= 0);
                            if (pdata->vecCards[c] == -1)
                                columns.Add(new U8Column(pdata, pdata->getters[c], c, name));
                            else
                                columns.Add(new VectorUInt8Column(pdata, pdata->getters[c], c, name, new VectorType(NumberType.U8, (int)pdata->vecCards[c])));
                            break;
                        case DataKind.I1:
                            if (pdata->vecCards[c] == -1)
                                columns.Add(new I1Column(pdata, pdata->getters[c], c, name));
                            else
                                columns.Add(new VectorInt1Column(pdata, pdata->getters[c], c, name, new VectorType(NumberType.I1, (int)pdata->vecCards[c])));
                            break;
                        case DataKind.I2:
                            if (pdata->vecCards[c] == -1)
                                columns.Add(new I2Column(pdata, pdata->getters[c], c, name));
                            else
                                columns.Add(new VectorInt2Column(pdata, pdata->getters[c], c, name, new VectorType(NumberType.I2, (int)pdata->vecCards[c])));
                            break;
                        case DataKind.I4:
                            if (pdata->vecCards[c] == -1)
                                columns.Add(new I4Column(pdata, pdata->getters[c], c, name));
                            else
                                columns.Add(new VectorInt4Column(pdata, pdata->getters[c], c, name, new VectorType(NumberType.I4, (int)pdata->vecCards[c])));
                            break;
                        case DataKind.I8:
                            if (pdata->vecCards[c] == -1)
                                columns.Add(new I8Column(pdata, pdata->getters[c], c, name));
                            else
                                columns.Add(new VectorInt8Column(pdata, pdata->getters[c], c, name, new VectorType(NumberType.I8, (int)pdata->vecCards[c])));
                            break;
                        case DataKind.R8:
                            if (pdata->vecCards[c] == -1)
                                columns.Add(new R8Column(pdata, pdata->getters[c], c, name));
                            else
                                columns.Add(new VectorR8Column(pdata, pdata->getters[c], c, name, new VectorType(NumberType.R8, (int)pdata->vecCards[c])));
                            break;
                        case DataKind.R4:
                            if (pdata->vecCards[c] == -1)
                                columns.Add(new R4Column(pdata, pdata->getters[c], c, name));
                            else
                                columns.Add(new VectorR4Column(pdata, pdata->getters[c], c, name, new VectorType(NumberType.R4, (int)pdata->vecCards[c])));
                            break;
                        case DataKind.Text:
                            columns.Add(new TextColumn(pdata, pdata->getters[c], c, name));
                            break;
                    }
                }

                _columns = columns.ToArray();
                var schemaBuilder = new SchemaBuilder();
                schemaBuilder.AddColumns(columns.Select(c => c.DetachedColumn));
                Schema = schemaBuilder.GetSchema();
            }

            public long? GetRowCount()
            {
                return _rowCount;
            }

            public RowCursor GetRowCursor(Func<int, bool> needCol, Random rand = null)
            {
                _host.CheckValue(needCol, nameof(needCol));
                _host.CheckValueOrNull(rand);

                var active = Utils.BuildArray(_columns.Length, needCol);
                return NativeRowCursor.CreateSet(_host, this, active, 1, rand)[0];
            }

            public RowCursor[] GetRowCursorSet(Func<int, bool> needCol, int n, Random rand = null)
            {
                _host.CheckValue(needCol, nameof(needCol));
                _host.CheckValueOrNull(rand);

                var active = Utils.BuildArray(_columns.Length, needCol);
                return NativeRowCursor.CreateSet(_host, this, active, n, rand);
            }

            public void Dispose()
            {
                foreach (var col in _columns)
                    col.Dispose();
            }

            private static bool GetLabels(DataSourceBlock* pdata, int colIndex, int count, ref VBuffer<ReadOnlyMemory<char>> buffer)
            {
                Contracts.Assert(count >= 0);
                if (count <= 0)
                {
                    buffer = VBufferEditor.Create(ref buffer, 0, 0).Commit();
                    return false;
                }

                var keyNamesGetter = MarshalDelegate<KeyNamesGetter>(pdata->labelsGetter);

                sbyte*[] buf = new sbyte*[count];
                fixed (sbyte** p = buf)
                {
                    if (!keyNamesGetter(pdata, colIndex, count, p))
                    {
                        buffer = VBufferEditor.Create(ref buffer, 0, 0).Commit();
                        return false;
                    }

                    var editor = VBufferEditor.Create(ref buffer, count);
                    for (int i = 0; i < count; i++)
                        Bridge.BytesToText(p[i], ref editor.Values[i]);

                    buffer = editor.Commit();
                }
                return true;
            }

            private sealed class NativeRowCursor : RootCursorBase
            {
                private readonly NativeDataView _view;
                private readonly TextColumnReader _reader;
                private readonly bool[] _active;
                private long _batchId;
                private Batch _batchTxt;
                private bool _justLoaded;
                private bool _disposed;

                public override Schema Schema => _view.Schema;

                public override long Batch => _batchId;

                private NativeRowCursor(IChannelProvider provider, NativeDataView view, bool[] active, Random rand, TextColumnReader reader)
                    : base(provider)
                {
                    Contracts.AssertValue(provider);
                    provider.AssertValue(view);
                    provider.AssertValue(active);
                    provider.AssertValueOrNull(rand);

                    _reader = reader;
                    _view = view;
                    _active = active;
                    _batchId = -1;
                    _batchTxt = default(Batch);
                    _justLoaded = false;
                }

                public override ValueGetter<TValue> GetGetter<TValue>(int col)
                {
                    Ch.CheckParam(_active[col], nameof(col), "column is not active");
                    var column = _view._columns[col] as Column<TValue>;
                    if (column == null)
                        throw Ch.Except("Invalid TValue: '{0}'", typeof(TValue));

                    return
                        (ref TValue value) =>
                        {
                            Ch.Check(IsGood);
                            long index = Position % BatchSize + _batchId * BatchSize;
                            Ch.Assert(0 <= index && index < _view._rowCount);
                            column.CopyOut(index, _batchTxt, ref value);
                        };
                }

                public override bool IsColumnActive(int col)
                {
                    Contracts.Check(0 <= col && col < Schema.Count);
                    return _active[col];
                }

                protected override void Dispose(bool disposing)
                {
                    if (_disposed)
                        return;

                    _disposed = true;
                    _reader.Release();
                    base.Dispose(disposing);
                }

                public override ValueGetter<RowId> GetIdGetter()
                {
                    return
                        (ref RowId val) =>
                        {
                            Ch.Check(IsGood, "Cannot call ID getter in current state");
                            long index = Position % BatchSize + _batchId * BatchSize;
                            val = new RowId((ulong)index, 0);
                        };
                }

                protected override bool MoveNextCore()
                {
                    Ch.Assert(State != CursorState.Done);
                    long index = Position % BatchSize + _batchId * BatchSize;
                    Ch.Assert(index < _view._rowCount);
                    if ((Position + 1) % BatchSize == 0 && !_justLoaded)
                    {
                        _batchTxt = _reader.GetBatch();
                        if (_batchTxt.Equals(default(Batch)))
                            return false;
                        _batchId = _batchTxt.BatchId;
                        _justLoaded = true;
                    }
                    else if ((Position + 1) % BatchSize != 0)
                        _justLoaded = false;
                    index = (Position + 1) % BatchSize + _batchId * BatchSize;

                    return index < _view._rowCount;
                }

                public static RowCursor[] CreateSet(IChannelProvider provider, NativeDataView view, bool[] active, int n, Random rand)
                {
                    Contracts.AssertValue(provider);
                    provider.AssertValue(view);
                    provider.AssertValue(active);
                    provider.AssertValueOrNull(rand);

                    var reader = new TextColumnReader(BatchSize, view._rowCount, n, view._columns);
                    if (n <= 1)
                    {
                        return new RowCursor[1] { new NativeRowCursor(provider, view, active, rand, reader) };
                    }

                    var cursors = new RowCursor[n];
                    try
                    {
                        for (int i = 0; i < cursors.Length; i++)
                            cursors[i] = new NativeRowCursor(provider, view, active, rand, reader);
                        var result = cursors;
                        cursors = null;
                        return result;
                    }
                    finally
                    {
                        if (cursors != null)
                        {
                            foreach (var curs in cursors)
                            {
                                if (curs != null)
                                    curs.Dispose();
                            }
                        }
                    }
                }
            }

            private class Row
            {
                // Currently contains only text data,
                // Potentially for better performance might contain all types of data.
                // Index is ColId as set in NativeDataView.
                public readonly ReadOnlyMemory<char>[] Line;

                public Row(int columns)
                {
                    Line = new ReadOnlyMemory<char>[columns];
                }
            }

            private struct Batch
            {
                // Total lines, up to the first line of this batch.
                public readonly long Total;
                public readonly long BatchId;
                public readonly Row[] Infos;
                public readonly Exception Exception;

                public Batch(long total, long batchId, Row[] infos)
                {
                    Contracts.AssertValueOrNull(infos);
                    Total = total;
                    BatchId = batchId;
                    Infos = infos;
                    Exception = null;
                }

                public Batch(Exception ex)
                {
                    Contracts.AssertValue(ex);
                    Total = 0;
                    BatchId = 0;
                    Infos = null;
                    Exception = ex;
                }
            }


            // This reads batches of text columns on a separate thread. It does UTF8 to UTF16 conversion.
            private sealed class TextColumnReader : IDisposable
            {
                private const int TimeoutMs = 10;
                private const int QueueSize = 1000;
                private readonly long _rowsCount;
                private readonly int _batchSize;
                private readonly Column[] _columns;

                // Ordered waiters on block numbers.
                // _waiterPublish orders publishing to _queue.
                private readonly OrderedWaiter _waiterPublish;

                // The reader can be referenced by multiple workers. This is the reference count.
                private int _cref;
                private BlockingCollection<Batch> _queue;
                private Thread _thdRead;
                private volatile bool _abort;

                public TextColumnReader(int batchSize, long rowsCount, int cref, Column[] columns)
                {
                    Contracts.Assert(batchSize >= 2);
                    Contracts.Assert(rowsCount >= 0);
                    Contracts.Assert(cref > 0);

                    _columns = columns;
                    _rowsCount = rowsCount;
                    _batchSize = batchSize;
                    _cref = cref;

                    _waiterPublish = new OrderedWaiter(firstCleared: true);

                    _queue = new BlockingCollection<Batch>(QueueSize);
                    _thdRead = Utils.CreateBackgroundThread(ThreadProc);
                    _thdRead.Start();
                }

                public void Release()
                {
                    int n = Interlocked.Decrement(ref _cref);
                    Contracts.Assert(n >= 0);

                    if (n != 0)
                        return;

                    if (_thdRead != null)
                    {
                        _abort = true;
                        _waiterPublish.IncrementAll();
                        _thdRead.Join();
                        _thdRead = null;
                    }

                    if (_queue != null)
                    {
                        _queue.Dispose();
                        _queue = null;
                    }
                }

                public Batch GetBatch()
                {
                    Exception inner = null;
                    try
                    {
                        var batch = _queue.Take();
                        if (batch.Exception == null)
                            return batch;
                        inner = batch.Exception;
                    }
                    catch (InvalidOperationException)
                    {
                        if (_queue.IsAddingCompleted)
                            return default(Batch);
                        throw;
                    }
                    Contracts.AssertValue(inner);
                    throw Contracts.ExceptDecode(inner, "Stream reading encountered exception");
                }

                private void ThreadProc()
                {
                    Contracts.Assert(_batchSize >= 2);

                    try
                    {
                        if (_rowsCount <= 0)
                            return;

                        long batchId = -1;
                        long total = 0;
                        var txtColumns = _columns.Where(c => c.DetachedColumn.Type is TextType).ToList();
                        int index = 0;
                        var infos = new Row[_batchSize];

                        for (long irow = 0; irow < _rowsCount; irow++)
                        {
                            var line = new Row(_columns.Length);
                            foreach (var column in txtColumns)
                            {
                                if (_abort)
                                    return;

                                ReadOnlyMemory<char> text = default(ReadOnlyMemory<char>);
                                var col = column as TextColumn;
                                col.CopyOutEx(irow, ref text);
                                line.Line[col.ColIndex] = text;
                            }

                            infos[index] = line;
                            if (++index >= infos.Length)
                            {
                                batchId++;
                                var lines = new Batch(total - index + 1, batchId, infos);
                                while (!_queue.TryAdd(lines, TimeoutMs))
                                {
                                    if (_abort)
                                        return;
                                }
                                infos = new Row[_batchSize];
                                index = 0;
                            }
                            if (++total >= _rowsCount)
                            {
                                PostPartial(total - index, ref batchId, index, infos);
                                return;
                            }
                        }
                    }
                    catch (Exception ex)
                    {
                        while (!_queue.TryAdd(new Batch(ex), TimeoutMs))
                        {
                            if (_abort)
                                return;
                        }
                    }
                    finally
                    {
                        _queue.CompleteAdding();
                    }
                }

                private void PostPartial(long total, ref long batchId, int index, Row[] infos)
                {
                    Contracts.Assert(0 <= total);
                    Contracts.Assert(0 <= index && index < Utils.Size(infos));

                    // Queue the last partial batch.
                    if (index <= 0)
                        return;

                    Array.Resize(ref infos, index);
                    batchId++;
                    while (!_queue.TryAdd(new Batch(total, batchId, infos), TimeoutMs))
                    {
                        if (_abort)
                            return;
                    }
                }

                public void Dispose()
                {
                    if(_queue != null)
                        _queue.Dispose();
                }
            }

            #region Column implementations

            private abstract class Column : IDisposable
            {
                protected DataSourceBlock* Data;
                public readonly int ColIndex;
                protected const string AlreadyDisposed = "Native wrapped column has been disposed";

                protected Column(DataSourceBlock* data, int colIndex, string name, ColumnType type)
                {
                    Contracts.AssertNonWhiteSpace(name);
                    Contracts.AssertValue(type);
                    Data = data;
                    ColIndex = colIndex;
                    DetachedColumn = new Schema.DetachedColumn(name, type);
                }

                public virtual void Dispose()
                {
                    Data = null;
                }

                /// This field contains some duplicate information with <see cref="Schema">.
                /// For more information please see the remarks on <see cref="Schema">.
                public Schema.DetachedColumn DetachedColumn { get; protected set; }
            }

            private abstract class Column<TOut> : Column
            {
                protected Column(DataSourceBlock* data, int colIndex, string name, ColumnType type)
                    : base(data, colIndex, name, type)
                {
                    Contracts.Assert(typeof(TOut) == type.RawType);
                }

                /// <summary>
                /// Produce the output value given the index.
                /// </summary>
                public abstract void CopyOut(long index, Batch batch, ref TOut value);
            }

            private sealed class BoolColumn : Column<bool>
            {
                private BLGetter _getter;

                public BoolColumn(DataSourceBlock* data, void* getter, int colIndex, string name)
                    : base(data, colIndex, name, BoolType.Instance)
                {
                    _getter = MarshalDelegate<BLGetter>(getter);
                }

                public override void CopyOut(long index, Batch batch, ref bool value)
                {
                    Contracts.Check(Data != null, AlreadyDisposed);
                    Contracts.Assert(0 <= index);
                    _getter(Data, ColIndex, index, out var val);
                    if (val < 0)
                        throw new InvalidOperationException(
                            $"Bool type doesnt support missing data, use floats or doubles." +
                            $" ColIndex: {ColIndex}, index: {index}");
                    value = val == 0 ? false : true;
                }

                public override void Dispose()
                {
                    _getter = null;
                    base.Dispose();
                }
            }

            private sealed class I1Column : Column<sbyte>
            {
                private I1Getter _getter;

                public I1Column(DataSourceBlock* data, void* getter, int colIndex, string name)
                    : base(data, colIndex, name, NumberType.I1)
                {
                    _getter = MarshalDelegate<I1Getter>(getter);
                }

                public override void CopyOut(long index, Batch batch, ref sbyte value)
                {
                    Contracts.Check(Data != null, AlreadyDisposed);
                    Contracts.Assert(0 <= index);
                    _getter(Data, ColIndex, index, out var val);
                    value = val;
                }

                public override void Dispose()
                {
                    _getter = null;
                    base.Dispose();
                }
            }

            private sealed class I2Column : Column<short>
            {
                private I2Getter _getter;

                public I2Column(DataSourceBlock* data, void* getter, int colIndex, string name)
                    : base(data, colIndex, name, NumberType.I2)
                {
                    _getter = MarshalDelegate<I2Getter>(getter);
                }

                public override void CopyOut(long index, Batch batch, ref short value)
                {
                    Contracts.Check(Data != null, AlreadyDisposed);
                    Contracts.Assert(0 <= index);
                    _getter(Data, ColIndex, index, out var val);
                    value = val;
                }

                public override void Dispose()
                {
                    _getter = null;
                    base.Dispose();
                }
            }

            private sealed class I4Column : Column<int>
            {
                private I4Getter _getter;

                public I4Column(DataSourceBlock* data, void* getter, int colIndex, string name)
                    : base(data, colIndex, name, NumberType.I4)
                {
                    _getter = MarshalDelegate<I4Getter>(getter);
                }

                public override void CopyOut(long index, Batch batch, ref int value)
                {
                    Contracts.Check(Data != null, AlreadyDisposed);
                    Contracts.Assert(0 <= index);
                    _getter(Data, ColIndex, index, out var val);
                    value = val;
                }

                public override void Dispose()
                {
                    _getter = null;
                    base.Dispose();
                }
            }

            private sealed class I8Column : Column<long>
            {
                private I8Getter _getter;

                public I8Column(DataSourceBlock* data, void* getter, int colIndex, string name)
                    : base(data, colIndex, name, NumberType.I8)
                {
                    _getter = MarshalDelegate<I8Getter>(getter);
                }

                public override void CopyOut(long index, Batch batch, ref long value)
                {
                    // REVIEW: There is a race condition if the data pointer becomes
                    // invalid between checking _disposed and accessing the data pointer.
                    Contracts.Check(Data != null, AlreadyDisposed);
                    Contracts.Assert(0 <= index);
                    _getter(Data, ColIndex, index, out var val);
                    value = val;
                }

                public override void Dispose()
                {
                    _getter = null;
                    base.Dispose();
                }
            }

            private sealed class U1Column : Column<byte>
            {
                private U1Getter _getter;

                public U1Column(DataSourceBlock* data, void* getter, int colIndex, string name)
                    : base(data, colIndex, name, NumberType.U1)
                {
                    _getter = MarshalDelegate<U1Getter>(getter);
                }

                public override void CopyOut(long index, Batch batch, ref byte value)
                {
                    Contracts.Check(Data != null, AlreadyDisposed);
                    Contracts.Assert(0 <= index);
                    _getter(Data, ColIndex, index, out value);
                }

                public override void Dispose()
                {
                    _getter = null;
                    base.Dispose();
                }
            }

            private sealed class U2Column : Column<ushort>
            {
                private U2Getter _getter;

                public U2Column(DataSourceBlock* data, void* getter, int colIndex, string name)
                    : base(data, colIndex, name, NumberType.U2)
                {
                    _getter = MarshalDelegate<U2Getter>(getter);
                }

                public override void CopyOut(long index, Batch batch, ref ushort value)
                {
                    Contracts.Check(Data != null, AlreadyDisposed);
                    Contracts.Assert(0 <= index);
                    _getter(Data, ColIndex, index, out value);
                }

                public override void Dispose()
                {
                    _getter = null;
                    base.Dispose();
                }
            }

            private sealed class U4Column : Column<uint>
            {
                private U4Getter _getter;

                public U4Column(DataSourceBlock* data, void* getter, int colIndex, string name)
                    : base(data, colIndex, name, NumberType.U4)
                {
                    _getter = MarshalDelegate<U4Getter>(getter);
                }

                public override void CopyOut(long index, Batch batch, ref uint value)
                {
                    Contracts.Check(Data != null, AlreadyDisposed);
                    Contracts.Assert(0 <= index);
                    _getter(Data, ColIndex, index, out value);
                }

                public override void Dispose()
                {
                    _getter = null;
                    base.Dispose();
                }
            }

            private sealed class U8Column : Column<ulong>
            {
                private U8Getter _getter;

                public U8Column(DataSourceBlock* data, void* getter, int colIndex, string name)
                    : base(data, colIndex, name, NumberType.U8)
                {
                    _getter = MarshalDelegate<U8Getter>(getter);
                }

                public override void CopyOut(long index, Batch batch, ref ulong value)
                {
                    // REVIEW: There is a race condition if the data pointer becomes
                    // invalid between checking _disposed and accessing the data pointer.
                    Contracts.Check(Data != null, AlreadyDisposed);
                    Contracts.Assert(0 <= index);
                    _getter(Data, ColIndex, index, out value);
                }

                public override void Dispose()
                {
                    _getter = null;
                    base.Dispose();
                }
            }

            private sealed class R8Column : Column<double>
            {
                private R8Getter _getter;

                public R8Column(DataSourceBlock* data, void* getter, int colIndex, string name)
                    : base(data, colIndex, name, NumberType.R8)
                {
                    _getter = MarshalDelegate<R8Getter>(getter);
                }

                public override void CopyOut(long index, Batch batch, ref double value)
                {
                    // REVIEW: There is a race condition if the data pointer becomes
                    // invalid between checking _disposed and accessing the data pointer.
                    Contracts.Check(Data != null, AlreadyDisposed);
                    Contracts.Assert(0 <= index);
                    _getter(Data, ColIndex, index, out value);
                }

                public override void Dispose()
                {
                    _getter = null;
                    base.Dispose();
                }
            }

            private sealed class R4Column : Column<float>
            {
                private R4Getter _getter;

                public R4Column(DataSourceBlock* data, void* getter, int colIndex, string name)
                    : base(data, colIndex, name, NumberType.R4)
                {
                    _getter = MarshalDelegate<R4Getter>(getter);
                }

                public override void CopyOut(long index, Batch batch, ref float value)
                {
                    Contracts.Check(Data != null, AlreadyDisposed);
                    Contracts.Assert(0 <= index);
                    _getter(Data, ColIndex, index, out value);
                }

                public override void Dispose()
                {
                    _getter = null;
                    base.Dispose();
                }
            }

            private sealed class TextColumn : Column<ReadOnlyMemory<char>>
            {
                private TXGetter _getter;

                public TextColumn(DataSourceBlock* data, void* getter, int colIndex, string name)
                    : base(data, colIndex, name, TextType.Instance)
                {
                    _getter = MarshalDelegate<TXGetter>(getter);
                }

                public void CopyOutEx(long index, ref ReadOnlyMemory<char> value)
                {
                    Contracts.Check(Data != null, AlreadyDisposed);
                    Contracts.Assert(0 <= index);
                    _getter(Data, ColIndex, index, out var pch, out int size, out int missing);
                    if (missing > 0)
                        value = ReadOnlyMemory<char>.Empty;
                    else if (size == -1) // need to do utf8 -> utf16 conversion
                        BytesToText((sbyte*)pch, ref value);
                    else
                        value = (new string(pch, 0, size)).AsMemory();
                }

                public override void CopyOut(long index, Batch batch, ref ReadOnlyMemory<char> value)
                {
                    Contracts.Check(Data != null, AlreadyDisposed);
                    Contracts.Assert(0 <= index);
                    Contracts.Assert(index >= batch.Total && index < batch.Total + batch.Infos.Length);
                    value = batch.Infos[index - batch.Total].Line[ColIndex];
                }

                public override void Dispose()
                {
                    _getter = null;
                    base.Dispose();
                }
            }

            // Find out if we need other kinds of keys.
            private sealed class KeyColumn : Column<uint>
            {
                private VBuffer<ReadOnlyMemory<char>> _keyValues;
                private U4Getter _getter;

                public KeyColumn(DataSourceBlock* data, void* getter, int colIndex, string name, int keyCount, ref VBuffer<ReadOnlyMemory<char>> keyValues)
                    : base(data, colIndex, name, new KeyType(DataKind.U4, keyCount))
                {
                    Contracts.Assert(keyCount >= 0);
                    Contracts.Assert(keyValues.Length == 0 || keyValues.Length == keyCount);

                    _getter = MarshalDelegate<U4Getter>(getter);

                    if (keyCount > 0 && keyCount == keyValues.Length)
                    {
                        keyValues.CopyTo(ref _keyValues);
                        ValueGetter<VBuffer<ReadOnlyMemory<char>>> getKeyValues =
                            (ref VBuffer<ReadOnlyMemory<char>> dst) => _keyValues.CopyTo(ref dst);
                        var metadataBuilder = new MetadataBuilder();
                        metadataBuilder.AddKeyValues(keyCount, TextType.Instance, getKeyValues);
                        DetachedColumn = new Schema.DetachedColumn(
                            name, new KeyType(DataKind.U4, keyCount), metadataBuilder.GetMetadata());
                    }
                }

                public override void CopyOut(long index, Batch batch, ref uint value)
                {
                    Contracts.Check(Data != null, AlreadyDisposed);
                    Contracts.Assert(0 <= index);
                    _getter(Data, ColIndex, index, out value);
                }

                public override void Dispose()
                {
                    _getter = null;
                    base.Dispose();
                }
            }

            private sealed class VectorBoolColumn : Column<VBuffer<bool>>
            {
                private BLVectorGetter _getter;
                private readonly int _length;

                public VectorBoolColumn(DataSourceBlock* data, void* getter, int colIndex, string name, VectorType type)
                    : base(data, colIndex, name, type)
                {
                    _getter = MarshalDelegate<BLVectorGetter>(getter);
                    _length = type.GetVectorSize();
                }

                public override void CopyOut(long index, Batch batch, ref VBuffer<bool> dst)
                {
                    Contracts.Check(Data != null, AlreadyDisposed);
                    Contracts.Assert(0 <= index);

                    _getter(Data, ColIndex, index, null, null, true, out var size);
                    var dstEditor = VBufferEditor.Create(ref dst, _length, size, requireIndicesOnDense: true);

                    if (size > 0)
                    {
                        fixed (int* pIndices = &dstEditor.Indices[0])
                        fixed (bool* pValues = &dstEditor.Values[0])
                        {
                            _getter(Data, ColIndex, index, pIndices, pValues, false, out size);
                        }
                    }

                    dst = dstEditor.Commit();
                }

                public override void Dispose()
                {
                    _getter = null;
                    base.Dispose();
                }
            }

            private sealed class VectorUInt1Column : Column<VBuffer<byte>>
            {
                private U1VectorGetter _getter;
                private readonly int _length;

                public VectorUInt1Column(DataSourceBlock* data, void* getter, int colIndex, string name, VectorType type)
                    : base(data, colIndex, name, type)
                {
                    _getter = MarshalDelegate<U1VectorGetter>(getter);
                    _length = type.GetVectorSize();
                }

                public override void CopyOut(long index, Batch batch, ref VBuffer<byte> dst)
                {
                    Contracts.Check(Data != null, AlreadyDisposed);
                    Contracts.Assert(0 <= index);

                    _getter(Data, ColIndex, index, null, null, true, out var size);
                    var dstEditor = VBufferEditor.Create(ref dst, _length, size, requireIndicesOnDense: true);

                    if (size > 0)
                    {
                        fixed (int* pIndices = &dstEditor.Indices[0])
                        fixed (byte* pValues = &dstEditor.Values[0])
                        {
                            _getter(Data, ColIndex, index, pIndices, pValues, false, out size);
                        }
                    }

                    dst = dstEditor.Commit();
                }

                public override void Dispose()
                {
                    _getter = null;
                    base.Dispose();
                }
            }

            private sealed class VectorUInt2Column : Column<VBuffer<ushort>>
            {
                private U2VectorGetter _getter;
                private readonly int _length;

                public VectorUInt2Column(DataSourceBlock* data, void* getter, int colIndex, string name, VectorType type)
                    : base(data, colIndex, name, type)
                {
                    _getter = MarshalDelegate<U2VectorGetter>(getter);
                    _length = type.GetVectorSize();
                }

                public override void CopyOut(long index, Batch batch, ref VBuffer<ushort> dst)
                {
                    Contracts.Check(Data != null, AlreadyDisposed);
                    Contracts.Assert(0 <= index);

                    _getter(Data, ColIndex, index, null, null, true, out var size);
                    var dstEditor = VBufferEditor.Create(ref dst, _length, size, requireIndicesOnDense: true);

                    if (size > 0)
                    {
                        fixed (int* pIndices = &dstEditor.Indices[0])
                        fixed (ushort* pValues = &dstEditor.Values[0])
                        {
                            _getter(Data, ColIndex, index, pIndices, pValues, false, out size);
                        }
                    }

                    dst = dstEditor.Commit();
                }

                public override void Dispose()
                {
                    _getter = null;
                    base.Dispose();
                }
            }

            private sealed class VectorUInt4Column : Column<VBuffer<uint>>
            {
                private U4VectorGetter _getter;
                private readonly int _length;

                public VectorUInt4Column(DataSourceBlock* data, void* getter, int colIndex, string name, VectorType type)
                    : base(data, colIndex, name, type)
                {
                    _getter = MarshalDelegate<U4VectorGetter>(getter);
                    _length = type.GetVectorSize();
                }

                public override void CopyOut(long index, Batch batch, ref VBuffer<uint> dst)
                {
                    Contracts.Check(Data != null, AlreadyDisposed);
                    Contracts.Assert(0 <= index);

                    _getter(Data, ColIndex, index, null, null, true, out var size);
                    var dstEditor = VBufferEditor.Create(ref dst, _length, size, requireIndicesOnDense: true);

                    if (size > 0)
                    {
                        fixed (int* pIndices = &dstEditor.Indices[0])
                        fixed (uint* pValues = &dstEditor.Values[0])
                        {
                            _getter(Data, ColIndex, index, pIndices, pValues, false, out size);
                        }
                    }

                    dst = dstEditor.Commit();
                }

                public override void Dispose()
                {
                    _getter = null;
                    base.Dispose();
                }
            }

            private sealed class VectorUInt8Column : Column<VBuffer<ulong>>
            {
                private U8VectorGetter _getter;
                private readonly int _length;

                public VectorUInt8Column(DataSourceBlock* data, void* getter, int colIndex, string name, VectorType type)
                    : base(data, colIndex, name, type)
                {
                    _getter = MarshalDelegate<U8VectorGetter>(getter);
                    _length = type.GetVectorSize();
                }

                public override void CopyOut(long index, Batch batch, ref VBuffer<ulong> dst)
                {
                    Contracts.Check(Data != null, AlreadyDisposed);
                    Contracts.Assert(0 <= index);

                    _getter(Data, ColIndex, index, null, null, true, out var size);
                    var dstEditor = VBufferEditor.Create(ref dst, _length, size, requireIndicesOnDense: true);

                    if (size > 0)
                    {
                        fixed (int* pIndices = &dstEditor.Indices[0])
                        fixed (ulong* pValues = &dstEditor.Values[0])
                        {
                            _getter(Data, ColIndex, index, pIndices, pValues, false, out size);
                        }
                    }

                    dst = dstEditor.Commit();
                }

                public override void Dispose()
                {
                    _getter = null;
                    base.Dispose();
                }
            }

            private sealed class VectorInt1Column : Column<VBuffer<sbyte>>
            {
                private I1VectorGetter _getter;
                private readonly int _length;

                public VectorInt1Column(DataSourceBlock* data, void* getter, int colIndex, string name, VectorType type)
                    : base(data, colIndex, name, type)
                {
                    _getter = MarshalDelegate<I1VectorGetter>(getter);
                    _length = type.GetVectorSize();
                }

                public override void CopyOut(long index, Batch batch, ref VBuffer<sbyte> dst)
                {
                    Contracts.Check(Data != null, AlreadyDisposed);
                    Contracts.Assert(0 <= index);

                    _getter(Data, ColIndex, index, null, null, true, out var size);
                    var dstEditor = VBufferEditor.Create(ref dst, _length, size, requireIndicesOnDense: true);

                    if (size > 0)
                    {
                        fixed (int* pIndices = &dstEditor.Indices[0])
                        fixed (sbyte* pValues = &dstEditor.Values[0])
                        {
                            _getter(Data, ColIndex, index, pIndices, pValues, false, out size);
                        }
                    }

                    dst = dstEditor.Commit();
                }

                public override void Dispose()
                {
                    _getter = null;
                    base.Dispose();
                }
            }

            private sealed class VectorInt2Column : Column<VBuffer<short>>
            {
                private I2VectorGetter _getter;
                private readonly int _length;

                public VectorInt2Column(DataSourceBlock* data, void* getter, int colIndex, string name, VectorType type)
                    : base(data, colIndex, name, type)
                {
                    _getter = MarshalDelegate<I2VectorGetter>(getter);
                    _length = type.GetVectorSize();
                }

                public override void CopyOut(long index, Batch batch, ref VBuffer<short> dst)
                {
                    Contracts.Check(Data != null, AlreadyDisposed);
                    Contracts.Assert(0 <= index);

                    _getter(Data, ColIndex, index, null, null, true, out var size);
                    var dstEditor = VBufferEditor.Create(ref dst, _length, size, requireIndicesOnDense: true);

                    if (size > 0)
                    {
                        fixed (int* pIndices = &dstEditor.Indices[0])
                        fixed (short* pValues = &dstEditor.Values[0])
                        {
                            _getter(Data, ColIndex, index, pIndices, pValues, false, out size);
                        }
                    }

                    dst = dstEditor.Commit();
                }

                public override void Dispose()
                {
                    _getter = null;
                    base.Dispose();
                }
            }

            private sealed class VectorInt4Column : Column<VBuffer<int>>
            {
                private I4VectorGetter _getter;
                private readonly int _length;

                public VectorInt4Column(DataSourceBlock* data, void* getter, int colIndex, string name, VectorType type)
                    : base(data, colIndex, name, type)
                {
                    _getter = MarshalDelegate<I4VectorGetter>(getter);
                    _length = type.GetVectorSize();
                }

                public override void CopyOut(long index, Batch batch, ref VBuffer<int> dst)
                {
                    Contracts.Check(Data != null, AlreadyDisposed);
                    Contracts.Assert(0 <= index);

                    _getter(Data, ColIndex, index, null, null, true, out var size);
                    var dstEditor = VBufferEditor.Create(ref dst, _length, size, requireIndicesOnDense: true);

                    if (size > 0)
                    {
                        fixed (int* pIndices = &dstEditor.Indices[0])
                        fixed (int* pValues = &dstEditor.Values[0])
                        {
                            _getter(Data, ColIndex, index, pIndices, pValues, false, out size);
                        }
                    }

                    dst = dstEditor.Commit();
                }

                public override void Dispose()
                {
                    _getter = null;
                    base.Dispose();
                }
            }

            private sealed class VectorInt8Column : Column<VBuffer<long>>
            {
                private I8VectorGetter _getter;
                private readonly int _length;

                public VectorInt8Column(DataSourceBlock* data, void* getter, int colIndex, string name, VectorType type)
                    : base(data, colIndex, name, type)
                {
                    _getter = MarshalDelegate<I8VectorGetter>(getter);
                    _length = type.GetVectorSize();
                }

                public override void CopyOut(long index, Batch batch, ref VBuffer<long> dst)
                {
                    Contracts.Check(Data != null, AlreadyDisposed);
                    Contracts.Assert(0 <= index);

                    _getter(Data, ColIndex, index, null, null, true, out var size);
                    var dstEditor = VBufferEditor.Create(ref dst, _length, size, requireIndicesOnDense: true);

                    if (size > 0)
                    {
                        fixed (int* pIndices = &dstEditor.Indices[0])
                        fixed (long* pValues = &dstEditor.Values[0])
                        {
                            _getter(Data, ColIndex, index, pIndices, pValues, false, out size);
                        }
                    }

                    dst = dstEditor.Commit();
                }

                // REVIEW: remind me why we don’t do the standard Dispose pattern with protected override void Dispose(true)?
                public override void Dispose()
                {
                    _getter = null;
                    base.Dispose();
                }
            }

            private sealed class VectorR4Column : Column<VBuffer<float>>
            {
                private R4VectorGetter _getter;
                private readonly int _length;

                public VectorR4Column(DataSourceBlock* data, void* getter, int colIndex, string name, VectorType type)
                    : base(data, colIndex, name, type)
                {
                    _getter = MarshalDelegate<R4VectorGetter>(getter);
                    _length = type.GetVectorSize();
                }

                public override void CopyOut(long index, Batch batch, ref VBuffer<float> dst)
                {
                    Contracts.Check(Data != null, AlreadyDisposed);
                    Contracts.Assert(0 <= index);

                    _getter(Data, ColIndex, index, null, null, true, out var size);
                    var dstEditor = VBufferEditor.Create(ref dst, _length, size, requireIndicesOnDense: true);

                    if (size > 0)
                    {
                        fixed (int* pIndices = &dstEditor.Indices[0])
                        fixed (float* pValues = &dstEditor.Values[0])
                        {
                            _getter(Data, ColIndex, index, pIndices, pValues, false, out size);
                        }
                    }

                    dst = dstEditor.Commit();
                }

                public override void Dispose()
                {
                    _getter = null;
                    base.Dispose();
                }
            }

            private sealed class VectorR8Column : Column<VBuffer<double>>
            {
                private R8VectorGetter _getter;
                private readonly int _length;

                public VectorR8Column(DataSourceBlock* data, void* getter, int colIndex, string name, VectorType type)
                    : base(data, colIndex, name, type)
                {
                    _getter = MarshalDelegate<R8VectorGetter>(getter);
                    _length = type.GetVectorSize();
                }

                public override void CopyOut(long index, Batch batch, ref VBuffer<double> dst)
                {
                    Contracts.Check(Data != null, AlreadyDisposed);
                    Contracts.Assert(0 <= index);

                    _getter(Data, ColIndex, index, null, null, true, out var size);
                    var dstEditor = VBufferEditor.Create(ref dst, _length, size, requireIndicesOnDense: true);

                    if (size > 0)
                    {
                        fixed (int* pIndices = &dstEditor.Indices[0])
                        fixed (double* pValues = &dstEditor.Values[0])
                        {
                            _getter(Data, ColIndex, index, pIndices, pValues, false, out size);
                        }
                    }

                    dst = dstEditor.Commit();
                }

                public override void Dispose()
                {
                    _getter = null;
                    base.Dispose();
                }
            }

            #endregion
        }
    }
}