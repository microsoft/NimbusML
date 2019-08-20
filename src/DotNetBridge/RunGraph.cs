//------------------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//------------------------------------------------------------------------------

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using Microsoft.DataPrep.Common;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Data.IO;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace Microsoft.MachineLearning.DotNetBridge
{
    public unsafe static partial class Bridge
    {
        // std:null specifier in a graph, used to redirect output to std::null
        const string STDNULL = "<null>";

        private sealed class RunGraphArgs
        {
#pragma warning disable 649 // never assigned
            [Argument(ArgumentType.AtMostOnce)]
            public string graph;
#pragma warning restore 649 // never assigned
        }

        private static void SaveIdvToFile(IDataView idv, string path, IHost host)
        {
            if (path == STDNULL)
                return;
            var extension = Path.GetExtension(path);
            IDataSaver saver;
            if (extension != ".csv" && extension != ".tsv" && extension != ".txt")
                saver = new BinarySaver(host, new BinarySaver.Arguments());
            else
            {
                var saverArgs = new TextSaver.Arguments
                {
                    OutputHeader = true,
                    OutputSchema = true,
                    Dense = true,

                    Separator = extension == ".csv" ? "comma" : "tab"
                };
                saver = new TextSaver(host, saverArgs);
            }

            using (var fs = File.OpenWrite(path))
            {
                saver.SaveData(fs, idv, Utils.GetIdentityPermutation(idv.Schema.Count)
                    .Where(x => !idv.Schema[x].IsHidden && saver.IsColumnSavable(idv.Schema[x].Type))
                    .ToArray());
            }
        }

        private static void SavePredictorModelToFile(PredictorModel model, string path, IHost host)
        {
            using (var fs = File.OpenWrite(path))
                model.Save(host, fs);
        }

        private static void RunGraphCore(EnvironmentBlock* penv, IHostEnvironment env, string graphStr, int cdata, DataSourceBlock** ppdata)
        {
            Contracts.AssertValue(env);

            var args = new RunGraphArgs();
            string err = null;
            if (!CmdParser.ParseArguments(env, graphStr, args, e => err = err ?? e))
                throw env.Except(err);

            var host = env.Register("RunGraph", penv->seed, null);

            JObject graph;
            try
            {
                graph = JObject.Parse(args.graph);
            }
            catch (JsonReaderException ex)
            {
                throw host.Except(ex, "Failed to parse experiment graph: {0}", ex.Message);
            }

            var runner = new GraphRunner(host, graph["nodes"] as JArray);

            var dvNative = new IDataView[cdata];
            try
            {
                for (int i = 0; i < cdata; i++)
                    dvNative[i] = new NativeDataView(host, ppdata[i]);

                // Setting inputs.
                var jInputs = graph["inputs"] as JObject;
                if (graph["inputs"] != null && jInputs == null)
                    throw host.Except("Unexpected value for 'inputs': {0}", graph["inputs"]);
                int iDv = 0;
                if (jInputs != null)
                {
                    foreach (var kvp in jInputs)
                    {
                        var pathValue = kvp.Value as JValue;
                        if (pathValue == null)
                            throw host.Except("Invalid value for input: {0}", kvp.Value);

                        var path = pathValue.Value<string>();
                        var varName = kvp.Key;
                        var type = runner.GetPortDataKind(varName);

                        switch (type)
                        {
                            case TlcModule.DataKind.FileHandle:
                                var fh = new SimpleFileHandle(host, path, false, false);
                                runner.SetInput(varName, fh);
                                break;
                            case TlcModule.DataKind.DataView:
                                IDataView dv;
                                if (!string.IsNullOrWhiteSpace(path))
                                {
                                    var extension = Path.GetExtension(path);
                                    if (extension == ".txt")
                                        dv = TextLoader.LoadFile(host, new TextLoader.Options(), new MultiFileSource(path));
                                    else if (extension == ".dprep")
                                        dv = LoadDprepFile(BytesToString(penv->pythonPath), path);
                                    else
                                        dv = new BinaryLoader(host, new BinaryLoader.Arguments(), path);
                                }
                                else
                                {
                                    Contracts.Assert(iDv < dvNative.Length);
                                    // prefetch all columns
                                    dv = dvNative[iDv++];
                                    var prefetch = new int[dv.Schema.Count];
                                    for (int i = 0; i < prefetch.Length; i++)
                                        prefetch[i] = i;
                                    dv = new CacheDataView(host, dv, prefetch);
                                }
                                runner.SetInput(varName, dv);
                                break;
                            case TlcModule.DataKind.PredictorModel:
                                PredictorModel pm;
                                if (!string.IsNullOrWhiteSpace(path))
                                {
                                    using (var fs = File.OpenRead(path))
                                        pm = new PredictorModelImpl(host, fs);
                                }
                                else
                                    throw host.Except("Model must be loaded from a file");
                                runner.SetInput(varName, pm);
                                break;
                            case TlcModule.DataKind.TransformModel:
                                TransformModel tm;
                                if (!string.IsNullOrWhiteSpace(path))
                                {
                                    using (var fs = File.OpenRead(path))
                                        tm = new TransformModelImpl(host, fs);
                                }
                                else
                                    throw host.Except("Model must be loaded from a file");
                                runner.SetInput(varName, tm);
                                break;
                            default:
                                throw host.Except("Port type {0} not supported", type);
                        }
                    }
                }
                runner.RunAll();

                // Reading outputs. 
                using (var ch = host.Start("Reading outputs"))
                {
                    var jOutputs = graph["outputs"] as JObject;
                    if (jOutputs != null)
                    {
                        foreach (var kvp in jOutputs)
                        {
                            var pathValue = kvp.Value as JValue;
                            if (pathValue == null)
                                throw host.Except("Invalid value for input: {0}", kvp.Value);
                            var path = pathValue.Value<string>();
                            var varName = kvp.Key;
                            var type = runner.GetPortDataKind(varName);

                            switch (type)
                            {
                                case TlcModule.DataKind.FileHandle:
                                    var fh = runner.GetOutput<IFileHandle>(varName);
                                    throw host.ExceptNotSupp("File handle outputs not yet supported.");
                                case TlcModule.DataKind.DataView:
                                    var idv = runner.GetOutput<IDataView>(varName);
                                    if (!string.IsNullOrWhiteSpace(path))
                                    {
                                        SaveIdvToFile(idv, path, host);
                                    }
                                    else
                                    {
                                        var infos = ProcessColumns(ref idv, penv->maxSlots, host);
                                        SendViewToNative(ch, penv, idv, infos);
                                    }
                                    break;
                                case TlcModule.DataKind.PredictorModel:
                                    var pm = runner.GetOutput<PredictorModel>(varName);
                                    if (!string.IsNullOrWhiteSpace(path))
                                    {
                                        SavePredictorModelToFile(pm, path, host);
                                    }
                                    else
                                        throw host.Except("Returning in-memory models is not supported");
                                    break;
                                case TlcModule.DataKind.TransformModel:
                                    var tm = runner.GetOutput<TransformModel>(varName);
                                    if (!string.IsNullOrWhiteSpace(path))
                                    {
                                        using (var fs = File.OpenWrite(path))
                                            tm.Save(host, fs);
                                    }
                                    else
                                        throw host.Except("Returning in-memory models is not supported");
                                    break;

                                case TlcModule.DataKind.Array:
                                    var objArray = runner.GetOutput<object[]>(varName);
                                    if (objArray is PredictorModel[])
                                    {
                                        var modelArray = (PredictorModel[])objArray;
                                        // Save each model separately
                                        for (var i = 0; i < modelArray.Length; i++)
                                        {
                                            var modelPath = string.Format(CultureInfo.InvariantCulture, path, i);
                                            SavePredictorModelToFile(modelArray[i], modelPath, host);
                                        }
                                    }
                                    else
                                        throw host.Except("DataKind.Array type {0} not supported", objArray.First().GetType());
                                    break;

                                default:
                                    throw host.Except("Port type {0} not supported", type);
                            }
                        }
                    }
                }
            }
            finally
            {
                // The raw data view is disposable so it lets go of unmanaged raw pointers before we return.
                for (int i = 0; i < dvNative.Length; i++)
                {
                    var view = dvNative[i];
                    if (view == null)
                        continue;
                    host.Assert(view is IDisposable);
                    var disp = (IDisposable)dvNative[i];
                    disp.Dispose();
                }
            }
        }

        private static IDataView LoadDprepFile(string pythonPath, string path)
        {
            DPrepSettings.Instance.PythonPath = pythonPath;
            return DataFlow.FromDPrepFile(path).ToDataView();
        }

        private static Dictionary<string, ColumnMetadataInfo> ProcessColumns(ref IDataView view, int maxSlots, IHostEnvironment env)
        {
            Dictionary<string, ColumnMetadataInfo> result = null;
            List<SlotsDroppingTransformer.ColumnOptions> drop = null;
            for (int i = 0; i < view.Schema.Count; i++)
            {
                if (view.Schema[i].IsHidden)
                    continue;

                var columnName = view.Schema[i].Name;
                var columnType = view.Schema[i].Type;
                if (columnType.IsKnownSizeVector())
                {
                    Utils.Add(ref result, columnName, new ColumnMetadataInfo(true, null, null));
                    if (maxSlots > 0 && columnType.GetValueCount() > maxSlots)
                    {
                        Utils.Add(ref drop,
                            new SlotsDroppingTransformer.ColumnOptions(
                                name: columnName,
                                slots: (maxSlots, null)));
                    }
                }
                else if (columnType is KeyDataViewType)
                {
                    Dictionary<uint, ReadOnlyMemory<char>> map = null;
                    if (columnType.GetKeyCount() > 0 && view.Schema[i].HasKeyValues())
                    {
                        var keyNames = default(VBuffer<ReadOnlyMemory<char>>);
                        view.Schema[i].Annotations.GetValue(AnnotationUtils.Kinds.KeyValues, ref keyNames);
                        map = keyNames.Items().ToDictionary(kv => (uint)kv.Key, kv => kv.Value);
                    }
                    Utils.Add(ref result, columnName, new ColumnMetadataInfo(false, null, map));
                }
            }

            if (drop != null)
            {
                var slotDropper = new SlotsDroppingTransformer(env, drop.ToArray());
                view = slotDropper.Transform(view);
            }

            return result;
        }
    }
}
