using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.DotNetBridge;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;

[assembly: LoadableClass(typeof(void), typeof(DotNetBridgeEntrypoints), null, typeof(SignatureEntryPointModule), "DotNetBridgeEntrypoints")]

[assembly: LoadableClass(VariableColumnTransform.Summary, typeof(VariableColumnTransform), null, typeof(SignatureLoadDataTransform),
    "", VariableColumnTransform.LoaderSignature)]

[assembly: LoadableClass(typeof(void), typeof(VariableColumnTransform), null, typeof(SignatureEntryPointModule), "VariableColumnTransform")]

namespace Microsoft.ML.DotNetBridge
{
    internal static class DotNetBridgeEntrypoints
    {
        [TlcModule.EntryPoint(Name = "Transforms.PrefixColumnConcatenator", Desc = ColumnConcatenatingTransformer.Summary,
            UserName = ColumnConcatenatingTransformer.UserName, ShortName = ColumnConcatenatingTransformer.LoadName)]
        public static CommonOutputs.TransformOutput ConcatColumns(IHostEnvironment env, ColumnCopyingTransformer.Options input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("PrefixConcatColumns");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            // Get all column names with preserving order.
            var colNames = new List<string>(input.Data.Schema.Count);
            for (int i = 0; i < input.Data.Schema.Count; i++)
                colNames.Add(input.Data.Schema[i].Name);

            // Iterate throuh input options, find matching source columns, create new input options
            var inputOptions = new ColumnConcatenatingTransformer.Options() { Data = input.Data };
            var columns = new List<ColumnConcatenatingTransformer.Column>(input.Columns.Length);
            foreach (var col in input.Columns)
            {
                var newCol = new ColumnConcatenatingTransformer.Column();
                newCol.Name = col.Name;
                var prefix = col.Source;
                newCol.Source = colNames.Where(x => x.StartsWith(prefix, StringComparison.InvariantCulture)).ToArray();
                if (newCol.Source.Length == 0)
                    throw new ArgumentOutOfRangeException("No matching columns found for prefix: " + prefix);

                columns.Add(newCol);
            }
            inputOptions.Columns = columns.ToArray();

            var xf = ColumnConcatenatingTransformer.Create(env, inputOptions, inputOptions.Data);
            return new CommonOutputs.TransformOutput { Model = new TransformModelImpl(env, xf, inputOptions.Data), OutputData = xf };
        }

        [TlcModule.EntryPoint(Name = "Transforms.VariableColumnTransform", Desc = VariableColumnTransform.Summary,
            UserName = "Variable Column Creator", ShortName = "Variable Column Creator")]
        public static CommonOutputs.TransformOutput CreateVariableColumn(IHostEnvironment env, VariableColumnTransform.Options inputOptions)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("VariableColumnCreator");
            EntryPointUtils.CheckInputArgs(host, inputOptions);

            var xf = VariableColumnTransform.Create(env, inputOptions, inputOptions.Data);
            return new CommonOutputs.TransformOutput { Model = new TransformModelImpl(env, xf, inputOptions.Data), OutputData = xf };
        }
    }
}
