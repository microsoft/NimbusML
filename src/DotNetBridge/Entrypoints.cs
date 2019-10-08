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

[assembly: LoadableClass(VariableColumnTransform.Summary, typeof(VariableColumnTransform), null, typeof(SignatureLoadDataTransform),
    "", VariableColumnTransform.LoaderSignature)]

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

        public sealed class ModelSchemaOutput
        {
            [TlcModule.Output(Desc = "The model schema", SortOrder = 1)]
            public IDataView Schema;
        }

        [TlcModule.EntryPoint(Name = "Models.Schema", Desc = "Retrieve output model schema")]
        public static ModelSchemaOutput GetSchema(IHostEnvironment env, TransformModelInput input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("GetSchema");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            return new ModelSchemaOutput { Schema = new EmptyDataView(host, input.Model.OutputSchema) };
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

        public sealed class ScoringTransformInput
        {
            [Argument(ArgumentType.Required, HelpText = "The dataset to be scored", SortOrder = 1)]
            public IDataView Data;

            [Argument(ArgumentType.Required, HelpText = "The predictor model to apply to data", SortOrder = 2)]
            public PredictorModel PredictorModel;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Suffix to append to the score columns", SortOrder = 3)]
            public string Suffix;
        }

        public sealed class ScoringTransformOutput
        {
            [TlcModule.Output(Desc = "The scored dataset", SortOrder = 1)]
            public IDataView ScoredData;

            [TlcModule.Output(Desc = "The scoring transform", SortOrder = 2)]
            public TransformModel ScoringTransform;
        }

        [TlcModule.EntryPoint(Name = "Transforms.DatasetScorerEx", Desc = "Score a dataset with a predictor model")]
        public static ScoringTransformOutput Score(IHostEnvironment env, ScoringTransformInput input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("ScoreModel");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            var inputData = input.Data;
            RoleMappedData data;
            IPredictor predictor;
            try
            {
                input.PredictorModel.PrepareData(host, inputData, out data, out predictor);
            }
            catch (Exception)
            {
                var inputColumnName = inputData.Schema[0].Name;
                predictor = input.PredictorModel.Predictor;
                var xf = new TypeConvertingTransformer(host, 
                    new TypeConvertingEstimator.ColumnOptions(inputColumnName, DataKind.Single, inputColumnName)).Transform(inputData);
                data = new RoleMappedData(xf, null, inputColumnName);
            }

            IDataView scoredPipe;
            using (var ch = host.Start("Creating scoring pipeline"))
            {
                ch.Trace("Creating pipeline");
                var bindable = ScoreUtils.GetSchemaBindableMapper(host, predictor);
                ch.AssertValue(bindable);

                var mapper = bindable.Bind(host, data.Schema);
                var scorer = ScoreUtils.GetScorerComponent(host, mapper, input.Suffix);
                scoredPipe = scorer.CreateComponent(host, data.Data, mapper, input.PredictorModel.GetTrainingSchema(host));
            }

            return
                new ScoringTransformOutput
                {
                    ScoredData = scoredPipe,
                    ScoringTransform = new TransformModelImpl(host, scoredPipe, inputData)
                };

        }
    }
}
