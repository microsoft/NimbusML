//------------------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//------------------------------------------------------------------------------

using System.IO;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints.JsonUtils;
using Microsoft.ML.Runtime.EntryPoints;
using Newtonsoft.Json;

namespace Microsoft.MachineLearning.ManifestGenerator
{
    public static class ManifestGenerator
    {
        public static void Main()
        {
            var env = new TlcEnvironment();
            var catalog = ModuleCatalog.CreateInstance(env);
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
