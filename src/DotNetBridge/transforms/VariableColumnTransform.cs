//------------------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//------------------------------------------------------------------------------

using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;
using Microsoft.ML.Internal.Utilities;


namespace Microsoft.ML.DotNetBridge
{
    using BitArray = System.Collections.BitArray;

    /// <summary>
    /// A transform that combines the specified input columns
    /// in to a single variable length vectorized column and
    /// passes the rest of the columns through unchanged.
    /// </summary>
    [BestFriend]
    internal sealed class VariableColumnTransform : IDataTransform, IRowToRowMapper
    {
        public class Options : TransformInputBase
        {
            [Argument(ArgumentType.Multiple, HelpText = "Features", SortOrder = 2)]
            public string[] Features;

            [Argument(ArgumentType.Multiple, HelpText = "Length Column Name", SortOrder = 2)]
            public string LengthColumnName;
        }

        private sealed class Bindings
        {
            public readonly List<int> outputToInputMap;
            public readonly List<int> vectorToInputMap;
            public int outputColumn;
            public int lengthColumn;

            public Bindings()
            {
                outputToInputMap = new List<int>();
                vectorToInputMap = new List<int>();
                outputColumn = -1;
                lengthColumn = -1;
            }
        }

        private readonly IHost _host;
        private readonly Bindings _bindings;
        private readonly HashSet<string> _columnNames;

        public IDataView Source { get; }

        DataViewSchema IRowToRowMapper.InputSchema => Source.Schema;

        private VariableColumnTransform(IHostEnvironment env, IDataView input, string[] features, string lengthColumnName)
        {
            Contracts.CheckValue(env, nameof(env));

            Source = input;
            _host = env.Register(RegistrationName);
            _bindings = new Bindings();

            _columnNames = (features == null) ? new HashSet<string>() :
                                                new HashSet<string>(features);

            OutputSchema = ProcessInputSchema(input.Schema, lengthColumnName);
        }

        internal const string Summary = "Combines the specified input columns in to a single variable length vectorized column.";

        public const string LoaderSignature = "VariableColumnTransform";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "VARLENCL",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(VariableColumnTransform).Assembly.FullName);
        }

        internal static string RegistrationName = "VariableColumnTransform";

        public static VariableColumnTransform Create(IHostEnvironment env, Options options, IDataView input)
        {
            return new VariableColumnTransform(env, input, options.Features, options.LengthColumnName);
        }

        public static VariableColumnTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register(RegistrationName);
            h.CheckValue(ctx, nameof(ctx));
            h.CheckValue(input, nameof(input));
            ctx.CheckAtModel(GetVersionInfo());
            return h.Apply("Loading Model", ch => new VariableColumnTransform(h, ctx, input));
        }

        private VariableColumnTransform(IHost host, ModelLoadContext ctx, IDataView input)
        {
            Contracts.AssertValue(host, nameof(host));
            host.CheckValue(input, nameof(input));

            Source = input;
            _host = host;

            // TODO: fill this in
        }

        void ICanSaveModel.Save(ModelSaveContext ctx)
        {
            _host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // TODO: fill this in
        }

        public bool CanShuffle => Source.CanShuffle;

        DataViewSchema IDataView.Schema => OutputSchema;
        public DataViewSchema OutputSchema { get; }

        private DataViewSchema ProcessInputSchema(DataViewSchema inputSchema, string lengthColumnName)
        {
            var builder = new DataViewSchema.Builder();
            for (int i = 0; i < inputSchema.Count; i++)
            {
                var name = inputSchema[i].Name;

                if (_columnNames.Contains(name))
                {
                    _bindings.vectorToInputMap.Add(i);
                }
                else if (name == lengthColumnName)
                {
                    _bindings.lengthColumn = i;
                }
                else
                {
                    builder.AddColumn(name, inputSchema[i].Type);
                    _bindings.outputToInputMap.Add(i);
                }
            }

            if (_bindings.vectorToInputMap.Count > 0)
            {
                var type = inputSchema[_bindings.vectorToInputMap[0]].Type as PrimitiveDataViewType;

                for (int i = 1; i < _bindings.vectorToInputMap.Count; i++)
                {
                    var nextType = inputSchema[_bindings.vectorToInputMap[i]].Type as PrimitiveDataViewType;
                    if (!nextType.Equals(type))
                    {
                        throw Contracts.Except("Input data types of the columns to vectorize must " +
                                               "all be of the same type. Found {0} and {1}.",
                                               type.ToString(),
                                               nextType.ToString());
                    }
                }

                var outputColumnType = new VectorDataViewType(type, 0);
                var outputColumnName = inputSchema[_bindings.vectorToInputMap[0]].Name;
                builder.AddColumn(outputColumnName, outputColumnType);

                _bindings.outputColumn = _bindings.outputToInputMap.Count;
            }

            return builder.ToSchema();
        }

        public long? GetRowCount()
        {
            return Source.GetRowCount();
        }

        public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null)
        {
            var predicate = RowCursorUtils.FromColumnsToPredicate(columnsNeeded, OutputSchema);

            _host.CheckValueOrNull(rand);
            return new Cursor(_host, this, _bindings, predicate, rand);
        }

        public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand = null)
        {
            var predicate = RowCursorUtils.FromColumnsToPredicate(columnsNeeded, OutputSchema);

            _host.CheckValueOrNull(rand);
            return new DataViewRowCursor[] { new Cursor(_host, this, _bindings, predicate, rand) };
        }

        private sealed class Cursor : RootCursorBase
        {
            private readonly IDataTransform _view;
            private readonly BitArray _active;
            private readonly Bindings _bindings;
            private readonly DataViewRowCursor _cursor;

            public override DataViewSchema Schema => _view.Schema;

            public override long Batch
            {
                get { return 0; }
            }

            public Cursor(IChannelProvider provider, IDataTransform view, Bindings bindings, Func<int, bool> predicate, Random rand)
                : base(provider)
            {
                Ch.AssertValue(view);
                Ch.AssertValueOrNull(rand);
                Ch.Assert(view.Schema.Count >= 0);

                _view = view;
                _bindings = bindings;
                _cursor = view.Source.GetRowCursorForAllColumns();
                _active = new BitArray(view.Schema.Count);

                if (predicate == null) _active.SetAll(true);
                else
                {
                    for (int i = 0; i < view.Schema.Count; ++i)
                        _active[i] = predicate(i);
                }
            }

            public override ValueGetter<DataViewRowId> GetIdGetter()
            {
                return (ref DataViewRowId val) =>
                {
                    Ch.Check(IsGood, RowCursorUtils.FetchValueStateError);
                    val = new DataViewRowId((ulong)Position, 0);
                };
            }

            public override bool IsColumnActive(DataViewSchema.Column column)
            {
                Ch.Check(column.Index < Schema.Count);
                return _active[column.Index];
            }

            private static readonly FuncInstanceMethodInfo1<Cursor, DataViewRow, Delegate> _makeVarLengthVectorGetterMethodInfo
                = FuncInstanceMethodInfo1<Cursor, DataViewRow, Delegate>.Create(target => target.MakeVarLengthVectorGetter<int>);

            private Delegate MakeVarLengthVectorGetter<T>(DataViewRow input)
            {
                var srcGetters = new ValueGetter<T>[_bindings.vectorToInputMap.Count];
                ValueGetter<long> lengthGetter = null;

                for (int i = 0; i < _bindings.vectorToInputMap.Count; i++)
                {
                    var column = input.Schema[_bindings.vectorToInputMap[i]];
                    srcGetters[i] = input.GetGetter<T>(column);
                }

                if (_bindings.lengthColumn >= 0)
                {
                    var column = input.Schema[_bindings.lengthColumn];
                    lengthGetter = input.GetGetter<long>(column);
                }

                T tmp = default(T);
                ValueGetter<VBuffer<T>> result = (ref VBuffer<T> dst) =>
                {
                    int length = _bindings.vectorToInputMap.Count;
                    if (lengthGetter != null)
                    {
                        long expectedLength = length;
                        lengthGetter(ref expectedLength);

                        if ((expectedLength >= 0) && (expectedLength < length))
                        {
                            length = (int)expectedLength;
                        }
                    }

                    var editor = VBufferEditor.Create(ref dst, length);

                    for (int i = 0; i < length; i++)
                    {
                        srcGetters[i](ref tmp);
                        editor.Values[i] = tmp;
                    }

                    dst = editor.Commit();
                };
                return result;
            }

            /// <summary>
            /// Returns a value getter delegate to fetch the value of column with the given columnIndex, from the row.
            /// This throws if the column is not active in this row, or if the type
            /// <typeparamref name="TValue"/> differs from this column's type.
            /// </summary>
            /// <typeparam name="TValue"> is the column's content type.</typeparam>
            /// <param name="column"> is the output column whose getter should be returned.</param>
            public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
            {
                if (column.Index == _bindings.outputColumn)
                {
                    VectorDataViewType columnType = column.Type as VectorDataViewType;
                    Delegate getter = Utils.MarshalInvoke(_makeVarLengthVectorGetterMethodInfo, this, columnType.ItemType.RawType, _cursor);
                    return getter as ValueGetter<TValue>;
                }
                else
                {
                    int inputIndex = _bindings.outputToInputMap[column.Index];
                    return _cursor.GetGetter<TValue>(_cursor.Schema[inputIndex]);
                }
            }

            protected override bool MoveNextCore()
            {
                return _cursor.MoveNext();
            }
        }

        /// <summary>
        /// Given a set of columns, return the input columns that are needed to generate those output columns.
        /// </summary>
        IEnumerable<DataViewSchema.Column> IRowToRowMapper.GetDependencies(IEnumerable<DataViewSchema.Column> dependingColumns)
            => dependingColumns;

        DataViewRow IRowToRowMapper.GetRow(DataViewRow input, IEnumerable<DataViewSchema.Column> activeColumns)
        {
            Contracts.CheckValue(input, nameof(input));
            Contracts.CheckValue(activeColumns, nameof(activeColumns));
            Contracts.CheckParam(input.Schema == Source.Schema, nameof(input), "Schema of input row must be the same as the schema the mapper is bound to");
            return input;
        }
    }
}