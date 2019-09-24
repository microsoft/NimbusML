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
    }
}
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.DotNetBridge;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;

[assembly: LoadableClass(typeof(void), typeof(DotNetBridgeEntrypoints), null, typeof(SignatureEntryPointModule), "DotNetBridgeEntrypoints")]

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

        public sealed class TransformModelInput
        {
            [Argument(ArgumentType.Required, HelpText = "The transform model.", SortOrder = 1)]
            public TransformModel Model;
        }

        [TlcModule.EntryPoint(Name = "Models.Schema", Desc = "Retrieve input and output model schemas")]
        public static Dictionary<string, string> GetSchema(IHostEnvironment env, TransformModelInput input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("GetSchema");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            var sbIn = new StringBuilder();
            foreach (var col in input.Model.InputSchema.Where(x => !x.IsHidden))
            {
                sbIn.Append(col.ToString());
                sbIn.Append(" ");
            }
            var sbOut = new StringBuilder();
            foreach (var col in input.Model.OutputSchema.Where(x => !x.IsHidden))
            {
                sbOut.Append(col.ToString());
                sbOut.Append(" ");
            }

            var result = new Dictionary<string, string>();
            result["input_schema"] = sbIn.ToString();
            result["output_schema"] = sbIn.ToString();
            return result;
        }
    }
}
