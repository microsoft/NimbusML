//------------------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//------------------------------------------------------------------------------

using System.IO;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints.JsonUtils;
using Microsoft.ML.Runtime.ImageAnalytics;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Runtime.LightGBM;
using Microsoft.ML.Runtime.Model.Onnx;
using Microsoft.ML.Runtime.PipelineInference;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.KMeans;
using Microsoft.ML.Trainers.PCA;
using Microsoft.ML.Trainers.SymSgd;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Categorical;
using Newtonsoft.Json;

namespace Microsoft.MachineLearning.ManifestGenerator
{
    public static class ManifestGenerator
    {
        public static void Main()
        {
            using (var env = new ConsoleEnvironment())
            {
                env.ComponentCatalog.RegisterAssembly(typeof(TextLoader).Assembly); // ML.Data
                env.ComponentCatalog.RegisterAssembly(typeof(LinearPredictor).Assembly); // ML.StandardLearners
                env.ComponentCatalog.RegisterAssembly(typeof(CategoricalTransform).Assembly); // ML.Transforms
                env.ComponentCatalog.RegisterAssembly(typeof(FastTreeBinaryPredictor).Assembly); // ML.FastTree
                env.ComponentCatalog.RegisterAssembly(typeof(KMeansPredictor).Assembly); // ML.KMeansClustering
                env.ComponentCatalog.RegisterAssembly(typeof(PcaPredictor).Assembly); // ML.PCA
                env.ComponentCatalog.RegisterAssembly(typeof(Experiment).Assembly); // ML.Legacy
                env.ComponentCatalog.RegisterAssembly(typeof(LightGbmBinaryPredictor).Assembly);
                env.ComponentCatalog.RegisterAssembly(typeof(TensorFlowTransform).Assembly);
                env.ComponentCatalog.RegisterAssembly(typeof(ImageLoaderTransform).Assembly);
                env.ComponentCatalog.RegisterAssembly(typeof(SymSgdClassificationTrainer).Assembly);
                env.ComponentCatalog.RegisterAssembly(typeof(AutoInference).Assembly);
                env.ComponentCatalog.RegisterAssembly(typeof(SaveOnnxCommand).Assembly);
                var catalog = env.ComponentCatalog;
                var jObj = JsonManifestUtils.BuildAllManifests(env, catalog);

                var jPath = "manifest.json";
                using (var file = File.OpenWrite(jPath))
                using (var writer = new StreamWriter(file))
                using (var jw = new JsonTextWriter(writer))
                {
                    jw.Formatting = Formatting.Indented;
                    jObj.WriteTo(jw);
                }
            }
        }
    }
}
