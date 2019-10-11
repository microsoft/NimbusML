//------------------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//------------------------------------------------------------------------------

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text.RegularExpressions;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Model.OnnxConverter;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.Ensemble;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.LightGbm;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.TimeSeries;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;


namespace Microsoft.ML.DotNetBridge
{
    public static class ManifestUtils
    {
        private static readonly Type[] _types = new Type[]
        {
            typeof(TextLoader),
            typeof(LinearModelParameters),
            typeof(OneHotEncodingTransformer),
            typeof(FastTreeBinaryModelParameters),
            typeof(EnsembleModelParameters),
            typeof(KMeansModelParameters),
            typeof(PcaModelParameters),
            typeof(CVSplit),
            typeof(LightGbmBinaryModelParameters),
            typeof(TensorFlowTransformer),
            typeof(ImageLoadingTransformer),
            typeof(SymbolicSgdLogisticRegressionBinaryTrainer),
            typeof(OnnxContext),
            typeof(SsaForecastingTransformer)
        };

        private static (IEnumerable<string> epListContents, JObject manifest) BuildManifests()
        {
            ConsoleEnvironment env = new ConsoleEnvironment();

            foreach (Type type in _types)
            {
                env.ComponentCatalog.RegisterAssembly(type.Assembly);
            }

            var catalog = env.ComponentCatalog;

            var regex = new Regex(@"\r\n?|\n", RegexOptions.Compiled);
            var epListContents = catalog.AllEntryPoints()
                .Select(x => string.Join("\t",
                x.Name,
                regex.Replace(x.Description, ""),
                x.Method.DeclaringType,
                x.Method.Name,
                x.InputType,
                x.OutputType)
                .Replace(Environment.NewLine, "", StringComparison.Ordinal))
                .OrderBy(x => x);

            var manifest = JsonManifestUtils.BuildAllManifests(env, catalog);

            //clean up the description from the new line characters
            if (manifest[FieldNames.TopEntryPoints] != null && manifest[FieldNames.TopEntryPoints] is JArray)
            {
                foreach (JToken entry in manifest[FieldNames.TopEntryPoints].Children())
                    if (entry[FieldNames.Desc] != null)
                        entry[FieldNames.Desc] = regex.Replace(entry[FieldNames.Desc].ToString(), "");
            }

            return (epListContents, manifest);
        }

        public static void ShowAssemblyInfo()
        {
            foreach (Type type in _types)
            {
                Assembly assembly = type.Assembly;
                Console.WriteLine(assembly.Location);
            }
        }

        public static void GenerateManifest(string filePath)
        {
            var (epListContents, jObj) = BuildManifests();

            if (!string.IsNullOrWhiteSpace(filePath))
                File.Delete(filePath);

            using (var file = File.OpenWrite(filePath))
            using (var writer = new StreamWriter(file))
            using (var jw = new JsonTextWriter(writer))
            {
                jw.Formatting = Formatting.Indented;
                jObj.WriteTo(jw);
            }
        }
    }
}
