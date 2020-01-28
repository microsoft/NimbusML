//------------------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//------------------------------------------------------------------------------

using System;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Featurizers;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.Ensemble;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.LightGbm;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.TimeSeries;

namespace Microsoft.ML.DotNetBridge
{
    /// <summary>
    /// The main entry point from native code. Note that GC / lifetime issues are critical to get correct.
    /// This code shares a bunch of information with native code via unsafe pointers. We need to carefully
    /// ensure that no native pointers are accessed after the invoked entry point returns. As an example
    /// of an implication of this: "returned" data is provided via a call back function (from managed to\
    /// native). It cannot simply be returned by the entry point since source data that the output data depends
    /// on is not accessible once we return from managed code.
    /// </summary>
    public unsafe static partial class Bridge
    {
        // For getting float values from NativeBridge.
        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        private unsafe delegate void R4Getter(DataSourceBlock* pv, int col, long index, out float dst);

        // For getting double values from NativeBridge.
        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        private unsafe delegate void R8Getter(DataSourceBlock* pv, int col, long index, out double dst);

        // For getting bool values from NativeBridge.
        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        private unsafe delegate void BLGetter(DataSourceBlock* pv, int col, long index, out sbyte dst);

        // For getting numpy.int8 values from NativeBridge.
        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        private unsafe delegate void I1Getter(DataSourceBlock* pv, int col, long index, out sbyte dst);

        // For getting numpy.int16 values from NativeBridge.
        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        private unsafe delegate void I2Getter(DataSourceBlock* pv, int col, long index, out short dst);

        // For getting numpy.int32 values from NativeBridge.
        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        private unsafe delegate void I4Getter(DataSourceBlock* pv, int col, long index, out int dst);

        // For getting numpy.int64 values from NativeBridge.
        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        private unsafe delegate void I8Getter(DataSourceBlock* pv, int col, long index, out long dst);

        // For getting numpy.uint8 values from NativeBridge.
        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        private unsafe delegate void U1Getter(DataSourceBlock* pv, int col, long index, out byte dst);

        // For getting numpy.uint16 values from NativeBridge.
        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        private unsafe delegate void U2Getter(DataSourceBlock* pv, int col, long index, out ushort dst);

        // For getting numpy.uint32 values from NativeBridge.
        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        private unsafe delegate void U4Getter(DataSourceBlock* pv, int col, long index, out uint dst);

        // For getting numpy.uint64 values from NativeBridge.
        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        private unsafe delegate void U8Getter(DataSourceBlock* pv, int col, long index, out ulong dst);

        // For getting string values from NativeBridge.
        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        private unsafe delegate void TXGetter(DataSourceBlock* pv, int col, long index, out char* pch, out int size, out int missing);

        // For getting key-type labels. id specifies the column id, count is the key type cardinality, and buffer
        // must be large enough for count pointers. Returns success.
        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        private unsafe delegate bool KeyNamesGetter(DataSourceBlock* pdata, int col, int count, sbyte** buffer);

        // For getting numpy.int64 vectors from NativeBridge.
        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        private unsafe delegate void I8VectorGetter(DataSourceBlock* pdata, int col, long index, int* indices, long* values, bool inquire, out int size);

        // For getting numpy.int32 vectors from NativeBridge.
        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        private unsafe delegate void I4VectorGetter(DataSourceBlock* pdata, int col, long index, int* indices, int* values, bool inquire, out int size);

        // For getting numpy.int16 vectors from NativeBridge.
        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        private unsafe delegate void I2VectorGetter(DataSourceBlock* pdata, int col, long index, int* indices, short* values, bool inquire, out int size);

        // For getting numpy.int8 vectors from NativeBridge.
        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        private unsafe delegate void I1VectorGetter(DataSourceBlock* pdata, int col, long index, int* indices, sbyte* values, bool inquire, out int size);

        // For getting numpy.uint64 vectors from NativeBridge.
        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        private unsafe delegate void U8VectorGetter(DataSourceBlock* pdata, int col, long index, int* indices, ulong* values, bool inquire, out int size);

        // For getting numpy.uint32 vectors from NativeBridge.
        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        private unsafe delegate void U4VectorGetter(DataSourceBlock* pdata, int col, long index, int* indices, uint* values, bool inquire, out int size);

        // For getting numpy.uint16 vectors from NativeBridge.
        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        private unsafe delegate void U2VectorGetter(DataSourceBlock* pdata, int col, long index, int* indices, ushort* values, bool inquire, out int size);

        // For getting numpy.uint8 vectors from NativeBridge.
        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        private unsafe delegate void U1VectorGetter(DataSourceBlock* pdata, int col, long index, int* indices, byte* values, bool inquire, out int size);

        // For getting numpy.bool vectors from NativeBridge.
        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        private unsafe delegate void BLVectorGetter(DataSourceBlock* pdata, int col, long index, int* indices, bool* values, bool inquire, out int size);

        // For getting float vectors from NativeBridge.
        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        private unsafe delegate void R4VectorGetter(DataSourceBlock* pdata, int col, long index, int* indices, float* values, bool inquire, out int size);

        // For getting double vectors from NativeBridge.
        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        private unsafe delegate void R8VectorGetter(DataSourceBlock* pdata, int col, long index, int* indices, double* values, bool inquire, out int size);

        // For setting bool values to NativeBridge.
        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        private unsafe delegate void BLSetter(EnvironmentBlock* penv, int col, long m, long n, byte value);

        // For setting float values to NativeBridge.
        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        private unsafe delegate void R4Setter(EnvironmentBlock* penv, int col, long m, long n, float value);

        // For setting double values to NativeBridge.
        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        private unsafe delegate void R8Setter(EnvironmentBlock* penv, int col, long m, long n, double value);

        // For setting I1 values to NativeBridge.
        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        private unsafe delegate void I1Setter(EnvironmentBlock* penv, int col, long m, long n, sbyte value);

        // For setting I2 values to NativeBridge.
        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        private unsafe delegate void I2Setter(EnvironmentBlock* penv, int col, long m, long n, short value);

        // For setting I4 values to NativeBridge.
        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        private unsafe delegate void I4Setter(EnvironmentBlock* penv, int col, long m, long n, int value);

        // For setting I8 values to NativeBridge.
        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        private unsafe delegate void I8Setter(EnvironmentBlock* penv, int col, long m, long n, long value);

        // For setting U1 values to NativeBridge.
        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        private unsafe delegate void U1Setter(EnvironmentBlock* penv, int col, long m, long n, byte value);

        // For setting U2 values to NativeBridge.
        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        private unsafe delegate void U2Setter(EnvironmentBlock* penv, int col, long m, long n, ushort value);

        // For setting U4 values to NativeBridge.
        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        private unsafe delegate void U4Setter(EnvironmentBlock* penv, int col, long m, long n, uint value);

        // For setting U8 values to NativeBridge.
        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        private unsafe delegate void U8Setter(EnvironmentBlock* penv, int col, long m, long n, ulong value);

        // For setting string values, to a generic pointer and index.
        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        private unsafe delegate void TXSetter(EnvironmentBlock* penv, int col, long m, long n, sbyte* pch, int cch);

        // For setting string key values, to a generic pointer and index.
        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        private unsafe delegate void KeyValueSetter(EnvironmentBlock* penv, int keyId, int keyCode, sbyte* pch, int cch);

        private enum FnId
        {
            HelloMlNet = 1,
            Generic = 2,
        }

        #region Callbacks to native

        // Call back to provide messages to native code.
        // REVIEW: Should we support embedded nulls? This API implies null termination.
        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        private unsafe delegate void MessageSink(EnvironmentBlock* penv, ChannelMessageKind kind, sbyte* sender, sbyte* message);

        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        private unsafe delegate void ModelSink(EnvironmentBlock* penv, byte* modelBytes, ulong modelSize);

        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        private unsafe delegate void DataSink(EnvironmentBlock* penv, DataViewBlock* pdata, out void** setters, out void* keyValueSetter);

        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        public unsafe delegate bool CheckCancelled();

        #endregion Callbacks to native

        /// <summary>
        /// This is provided by the native code. It provides general call backs for the task.
        /// Data source specific call backs are in another structure.
        /// </summary>
        [StructLayout(LayoutKind.Explicit)]
        private struct EnvironmentBlock
        {
#pragma warning disable 649 // never assigned
            [FieldOffset(0x00)]
            public readonly int verbosity;

            [FieldOffset(0x04)]
            public readonly int seed;

            // Call back to provide messages to native code.
            [FieldOffset(0x08)]
            public readonly void* messageSink;

            // Call back to provide data to native code.
            [FieldOffset(0x10)]
            public readonly void* dataSink;

            // Call back to provide model to native code.
            [FieldOffset(0x18)]
            public readonly void* modelSink;

            //Max slots to return for vector valued columns(<=0 to return all).
            [FieldOffset(0x20)]
            public readonly int maxSlots;

            // Call back to provide cancel flag.
            [FieldOffset(0x28)]
            public readonly void* checkCancel;

            // Path to python executable.
            [FieldOffset(0x30)]
            public readonly sbyte* pythonPath;
#pragma warning restore 649 // never assigned
        }

        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        private unsafe delegate int NativeGeneric(EnvironmentBlock* penv, sbyte* psz, int cdata, DataSourceBlock** ppdata);

        private static NativeGeneric FnGeneric;

        private static TDel MarshalDelegate<TDel>(void* pv)
        {
            Contracts.Assert(typeof(TDel).IsSubclassOf(typeof(Delegate)));
            Contracts.Assert(pv != null);
            return Marshal.GetDelegateForFunctionPointer<TDel>((IntPtr)pv);
        }

        /// <summary>
        /// This is the main FnGetter function. Given an FnId value, it returns a native-callable
        /// entry point address.
        /// </summary>
        private static unsafe IntPtr GetFn(FnId id)
        {
            switch (id)
            {
                default:
                    return default(IntPtr);
                case FnId.Generic:
                    if (FnGeneric == null)
                        Interlocked.CompareExchange(ref FnGeneric, GenericExec, null);
                    return Marshal.GetFunctionPointerForDelegate(FnGeneric);
            }
        }

        /// <summary>
        // The Generic entry point. The specific behavior is indicated in a string argument.
        /// </summary>
        private static unsafe int GenericExec(EnvironmentBlock* penv, sbyte* psz, int cdata, DataSourceBlock** ppdata)
        {
            var env = new RmlEnvironment(MarshalDelegate<CheckCancelled>(penv->checkCancel), penv->seed, verbose: penv != null && penv->verbosity > 3);
            var host = env.Register("ML.NET_Execution");

            env.ComponentCatalog.RegisterAssembly(typeof(TextLoader).Assembly); // ML.Data
            env.ComponentCatalog.RegisterAssembly(typeof(LinearModelParameters).Assembly); // ML.StandardLearners
            env.ComponentCatalog.RegisterAssembly(typeof(CategoricalCatalog).Assembly); // ML.Transforms
            env.ComponentCatalog.RegisterAssembly(typeof(FastTreeRegressionTrainer).Assembly); // ML.FastTree
            
            env.ComponentCatalog.RegisterAssembly(typeof(EnsembleModelParameters).Assembly); // ML.Ensemble
            env.ComponentCatalog.RegisterAssembly(typeof(KMeansModelParameters).Assembly); // ML.KMeansClustering
            env.ComponentCatalog.RegisterAssembly(typeof(PcaModelParameters).Assembly); // ML.PCA
            env.ComponentCatalog.RegisterAssembly(typeof(CVSplit).Assembly); // ML.EntryPoints

            env.ComponentCatalog.RegisterAssembly(typeof(OlsModelParameters).Assembly);
            env.ComponentCatalog.RegisterAssembly(typeof(LightGbmBinaryModelParameters).Assembly);
            env.ComponentCatalog.RegisterAssembly(typeof(TensorFlowTransformer).Assembly);
            //env.ComponentCatalog.RegisterAssembly(typeof(SymSgdClassificationTrainer).Assembly);
            //env.ComponentCatalog.RegisterAssembly(typeof(AutoInference).Assembly); // ML.PipelineInference
            env.ComponentCatalog.RegisterAssembly(typeof(DataViewReference).Assembly);
            env.ComponentCatalog.RegisterAssembly(typeof(ImageLoadingTransformer).Assembly);
            env.ComponentCatalog.RegisterAssembly(typeof(OnnxExportExtensions).Assembly);
            //env.ComponentCatalog.RegisterAssembly(typeof(TimeSeriesProcessingEntryPoints).Assembly);
            //env.ComponentCatalog.RegisterAssembly(typeof(ParquetLoader).Assembly);
            env.ComponentCatalog.RegisterAssembly(typeof(SsaChangePointDetector).Assembly);
            env.ComponentCatalog.RegisterAssembly(typeof(DotNetBridgeEntrypoints).Assembly);
            env.ComponentCatalog.RegisterAssembly(typeof(DateTimeTransformer).Assembly);

            using (var ch = host.Start("Executing"))
            {
                var sw = new System.Diagnostics.Stopwatch();
                sw.Start();
                try
                {
                    // code, pszIn, and pszOut can be null.
                    ch.Trace("Checking parameters");

                    host.CheckParam(penv != null, nameof(penv));
                    host.CheckParam(penv->messageSink != null, "penv->message");

                    host.CheckParam(psz != null, nameof(psz));

                    ch.Trace("Converting graph operands");
                    var graph = BytesToString(psz);

                    ch.Trace("Wiring message sink");
                    var message = MarshalDelegate<MessageSink>(penv->messageSink);
                    var messageValidator = new MessageValidator(host);
                    var lk = new object();
                    Action<IMessageSource, ChannelMessage> listener =
                        (sender, msg) =>
                        {
                            byte[] bs = StringToNullTerminatedBytes(sender.FullName);
                            string m = messageValidator.Validate(msg);
                            if (!string.IsNullOrEmpty(m))
                            {
                                byte[] bm = StringToNullTerminatedBytes(m);
                                lock (lk)
                                {
                                    fixed (byte* ps = bs)
                                    fixed (byte* pm = bm)
                                        message(penv, msg.Kind, (sbyte*)ps, (sbyte*)pm);
                                }
                            }
                        };
                    env.AddListener(listener);

                    host.CheckParam(cdata >= 0, nameof(cdata), "must be non-negative");
                    host.CheckParam(ppdata != null || cdata == 0, nameof(ppdata));
                    for (int i = 0; i < cdata; i++)
                    {
                        var pdata = ppdata[i];
                        host.CheckParam(pdata != null, "pdata");
                        host.CheckParam(0 <= pdata->ccol && pdata->ccol <= int.MaxValue, "ccol");
                        host.CheckParam(0 <= pdata->crow && pdata->crow <= long.MaxValue, "crow");
                        if (pdata->ccol > 0)
                        {
                            host.CheckParam(pdata->names != null, "names");
                            host.CheckParam(pdata->kinds != null, "kinds");
                            host.CheckParam(pdata->keyCards != null, "keyCards");
                            host.CheckParam(pdata->vecCards != null, "vecCards");
                            host.CheckParam(pdata->getters != null, "getters");
                        }
                    }

                    ch.Trace("Validating number of data sources");

                    // Wrap the data sets.
                    ch.Trace("Wrapping native data sources");
                    ch.Trace("Executing");
                    RunGraphCore(penv, host, graph, cdata, ppdata);
                }
                catch (Exception e)
                {
                    // Dump the exception chain.
                    var ex = e;
                    while (ex.InnerException != null)
                        ex = ex.InnerException;
                    ch.Error("*** {1}: '{0}'", ex.Message, ex.GetType());
                    return -1;
                }
                finally
                {
                    sw.Stop();
                    if (penv != null && penv->verbosity > 0)
                        ch.Info("Elapsed time: {0}", sw.Elapsed);
                    else
                        ch.Trace("Elapsed time: {0}", sw.Elapsed);
                }
            }
            return 0;
        }

        /// <summary>
        /// Convert UTF8 bytes with known length to ROM<char>. Negative length unsupported.
        /// </summary>
        internal static void BytesToText(sbyte* prgch, ulong bch, ref ReadOnlyMemory<char> dst)
        {
            if (bch > 0)
                dst = BytesToString(prgch, bch).AsMemory();
            else
                dst = ReadOnlyMemory<char>.Empty;
        }

        /// <summary>
        /// Convert null-terminated UTF8 bytes to ROM<char>. Null pointer unsupported.
        /// </summary>
        internal static void BytesToText(sbyte* psz, ref ReadOnlyMemory<char> dst)
        {
            if (psz != null)
                dst = BytesToString(psz).AsMemory();
            else
                dst = ReadOnlyMemory<char>.Empty;
        }

        /// <summary>
        /// Convert UTF8 bytes with known positive length to a string.
        /// </summary>
        internal static string BytesToString(sbyte* prgch, ulong bch)
        {
            Contracts.Assert(prgch != null);
            Contracts.Assert(bch > 0);
            return Encoding.UTF8.GetString((byte*)prgch, (int)bch);
        }

        /// <summary>
        /// Convert null-terminated UTF8 bytes to a string.
        /// </summary>
        internal static string BytesToString(sbyte* psz)
        {
            Contracts.Assert(psz != null);
            // REVIEW: Ideally should make this safer by always knowing the length.
            int cch = 0;
            while (psz[cch] != 0)
                cch++;

            if (cch == 0)
                return null;
            return Encoding.UTF8.GetString((byte*)psz, cch);
        }

        /// <summary>
        /// Convert a string to null-terminated UTF8 bytes.
        /// </summary>
        internal static byte[] StringToNullTerminatedBytes(string str)
        {
            // Note that it will result in multiple UTF-8 null bytes at the end, which is not ideal but harmless.
            return Encoding.UTF8.GetBytes(str + Char.MinValue);
        }
    }
}
