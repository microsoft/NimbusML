//------------------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//------------------------------------------------------------------------------

using System;
using System.Globalization;
using Microsoft.ML;

namespace Microsoft.MachineLearning.DotNetBridge
{
    /// <summary>
    /// This is a temporary solution to validate the messages from ML.NET to nimbusml.
    /// For every info, warning, trace and error message, validate with corresponding MessageRule[].
    /// </summary>
    public sealed class MessageValidator
    {
        private IHost _host;

        // List the rules for each message kind, _defaultRules apply to all messages.
        private static MessageRule[]  _warningMessageRules = new MessageRule[] {
                
            // Skip the warning:
            // No input ini specified. Raw Features will be used.
            // Warning:We seem to be processing a lot of data. Consider using the FastTree diskTranspose + (or dt + ) option, for slower but more memory efficient transposition. 
            new MessageRule(
                message =>
                {
                    return message.StartsWith(MlNetInterceptedMessages.NoInputIni, StringComparison.InvariantCulture) ||
                            message.Contains(MlNetInterceptedMessages.FastTreeDiskTranspose);
                },
                message =>
                {
                    return null;
                }
            ),

            // Fix the warning:
            // Automatically adding a MinMax normalization transform, 
            // use 'norm=Warn' or 'norm=No' to turn this behavior off
            new MessageRule(
                message =>
                {
                    return message.StartsWith(MlNetInterceptedMessages.AddingMinMax, StringComparison.InvariantCulture) &&
                            message.Contains(MlNetInterceptedMessages.NormWarn) &&
                            message.Contains(MlNetInterceptedMessages.NormNo);
                },
                message =>
                {
                    return "Automatically adding a MinMax normalization transform.";
                }
            ),

            // Skip the warning:
            // using -h 0 may be faster
            new MessageRule(
                message =>
                {
                    return string.Compare(message, MlNetInterceptedMessages.UseH, true, CultureInfo.InvariantCulture) == 0;
                },
                message =>
                {
                    return null;
                }
            ),

        };
        // Fix the error message:
        // Unrecognized command line argument...
        private static MessageRule[] _errorMessageRules = new MessageRule[]
        {
            new MessageRule(
                message =>
                {
                    return message.StartsWith(MlNetInterceptedMessages.MlNetUnrecognized, StringComparison.InvariantCulture);
                },
                message =>
                {
                    string arg;
                    string compName;
                    ExtractArgAndComponent(message, out arg, out compName);

                    string rmlCompName;
                    switch (compName)
                    {
                    case "Text":
                        rmlCompName = RmlComponents.Text;
                        break;
                    case "CategoricalTransform":
                        rmlCompName = RmlComponents.Cat;
                        break;
                    case "CategoricalHashTransform":
                        rmlCompName = RmlComponents.CatHash;
                        break;
                    case "Concat":
                        rmlCompName = RmlComponents.Concat;
                        break;
                    case "CountFeatureSelectionTransform":
                    case "MutualInformationFeatureSelectionTransform":
                        rmlCompName = RmlComponents.FeatureSelection;
                        break;
                    case "FastTreeBinaryClassification":
                    case "FastTreeRegression":
                        rmlCompName = RmlComponents.FastTree;
                        break;
                    case "SDCA":
                    case "SDCAR":
                        rmlCompName = RmlComponents.Sdca;
                        break;
                    case "FastForestClassification":
                    case "FastForestRegression":
                        rmlCompName = RmlComponents.FastForest;
                        break;
                    case "LogisticRegression":
                    case "MultiClassLogisticRegression":
                        rmlCompName = RmlComponents.LogisticRegression;
                        break;
                    case "BinaryNeuralNetwork":
                    case "MultiClassNeuralNetwork":
                    case "RegressionNeuralNetwork":
                        rmlCompName = RmlComponents.NeuralNet;
                        break;
                    case "OneClassSVM":
                        rmlCompName = RmlComponents.OneClassSvm;
                        break;
                    default:
                        rmlCompName = null;
                        break;
                    }

                    if (!string.IsNullOrEmpty(rmlCompName))
                        return string.Format(CultureInfo.InvariantCulture, "Invalid argument '{0}' in '{1}'", arg, rmlCompName);

                    return string.Format(CultureInfo.InvariantCulture, "Invalid argument'{0}'", arg);
                }),
            new MessageRule(
                message =>
                {
                    return message.StartsWith(MlNetInterceptedMessages.NoTrainingInstancesMessage, StringComparison.InvariantCulture) ||
                        message.StartsWith(MlNetInterceptedMessages.NoTrainingInstancesNNMessage, StringComparison.InvariantCulture);
                },
                message =>
                {
                    return "All rows in data has missing values (N/A).  Please clean missing data before training.";
                }),
            new MessageRule(
                message =>
                {
                    return message.Contains(MlNetInterceptedMessages.ConcatDifferentTypes);
                },
                message =>
                {
                    return DropMlNetTypes(message);
                }),
            new MessageRule(
                message => message.StartsWith(MlNetInterceptedMessages.ImageFeaturizerModelFile, StringComparison.InvariantCulture) || 
                            message.StartsWith(MlNetInterceptedMessages.ImageFeaturizerXmlFile, StringComparison.InvariantCulture),

                message => message.Replace(MlNetInterceptedMessages.MlNetImageFeaturizerLink, MmlImageFeaturizerLink)),
            new MessageRule(
                message => message.StartsWith(MlNetInterceptedMessages.SSWEFeaturizerModelFile, StringComparison.InvariantCulture),

                message => {
                return  "Error: *** Exception: 'SSWE model file is missing in folder."+
                        "You can download the missing model file."+
                        "Please check the following help page for more info: 'https://go.microsoft.com/fwlink/?linkid=838463'.'";
                }),
        };
        private static MessageRule[] _infoMessageRules = new MessageRule[] {
                
            // Fix the info:
            // LBFGS multi-threading will attempt to load dataset into memory. 
            // In case of out-of-memory issues, add 'numThreads=1' to the trainer 
            // arguments and 'cache=-' to the command line arguments to turn off multi-threading
            new MessageRule(
                message =>
                {
                    return message.StartsWith(MlNetInterceptedMessages.LbfgsMultiThreading, StringComparison.InvariantCulture) &&
                            message.Contains(MlNetInterceptedMessages.OneThread) &&
                            message.Contains(MlNetInterceptedMessages.NoCache);
                },
                message =>
                {
                    return "LBFGS multi-threading will attempt to load dataset into memory. " +
                            "In case of out-of-memory issues, turn off multi-threading by setting trainThreads to 1.";
                }
            ),
            // Fix the CUDA error message.
            new MessageRule(
                message =>
                {
                    return message.StartsWith(MlNetInterceptedMessages.CudaInitFail, StringComparison.InvariantCulture);
                },
                message =>
                {
                    return "Failed to initialize CUDA runtime. Possible reasons:" + "\n" +
                            @"1. The machine does not have CUDA-capable card. Supported devices have compute capability 2.0 and higher." + "\n" +
                            @"2. Outdated graphics drivers. Please install the latest drivers from http://www.nvidia.com/Drivers ." + "\n" +
                            @"3. CUDA runtime DLLs are missing, please see the GPU acceleration help for the installation instructions.";
                }
            )
        };
        private static MessageRule[] _traceMessageRules = Array.Empty<MessageRule>();
        private static MessageRule[] _defaultRules = Array.Empty<MessageRule>();

        public MessageValidator(IHost host)
        {
            _host = host;
        }

        /// <summary>
        /// Validate the message with corresponding rules. Only the first matched rule will be
        /// applied.
        /// </summary>
        /// <returns>Validated/fixed message text. Return null if the message should be skipped.</returns>
        public string Validate(ChannelMessage message)
        {
            _host.AssertNonEmpty(message.Message);

            string validatedMessage = message.Message;

            MessageRule[] rules = null;
            switch (message.Kind)
            {
                case ChannelMessageKind.Error:
                    rules = _errorMessageRules;
                    break;
                case ChannelMessageKind.Info:
                    rules = _infoMessageRules;
                    break;
                case ChannelMessageKind.Trace:
                    rules = _traceMessageRules;
                    break;
                case ChannelMessageKind.Warning:
                    rules = _warningMessageRules;
                    break;
                default:
                    _host.ExceptParam("ChannelMessage.Kind");
                    break;
            }

            bool matched = false;
            foreach (var rule in rules)
            {
                if (rule.IsMatch(validatedMessage))
                {
                    validatedMessage = rule.Decorate(validatedMessage);
                    matched = true;
                    break;
                }
            }

            if (!matched)
            {
                foreach (var rule in _defaultRules)
                {
                    if (rule.IsMatch(validatedMessage))
                    {
                        validatedMessage = rule.Decorate(validatedMessage);
                        break;
                    }
                }
            }

            return validatedMessage;
        }

        private static void ExtractArgAndComponent(string message, out string arg, out string comp)
        {
            var argNameLength = message.Substring(MlNetInterceptedMessages.MlNetUnrecognized.Length).IndexOf('\'');
            Contracts.Assert(argNameLength > 0);
            arg = message.Substring(MlNetInterceptedMessages.MlNetUnrecognized.Length, argNameLength);
            // The error message should contain this text, followed by the component's name,
            // and a list of valid arguments for Maml. We extract the component name and ignore the rest.
            var compNameStart = message.IndexOf(MlNetInterceptedMessages.UsageStartText, StringComparison.InvariantCulture) + MlNetInterceptedMessages.UsageStartText.Length;
            Contracts.Assert(compNameStart > MlNetInterceptedMessages.MlNetUnrecognized.Length + MlNetInterceptedMessages.UsageStartText.Length);
            var compNameLength = message.Substring(compNameStart).IndexOf('\'');
            comp = message.Substring(compNameStart, compNameLength);
        }

        private static string DropMlNetTypes(string message)
        {
            // Example message:
            // *** Exception: 'Column 'Features' has invalid source types: All source columns must have the same type. Source types: 'Vec<I4, 4>, Vec<R4, 8>, Vec<R4, 16>, Vec<R4, 14>'.
            const string BeginningOfConfusingPart = " Source types: '";
            int lim = message.IndexOf(BeginningOfConfusingPart, StringComparison.InvariantCulture);
            if (lim > 0)
                return message.Substring(0, lim);
            return message;
        }

        private static class MlNetInterceptedMessages
        {
            public static string MlNetUnrecognized = "*** Exception: 'Unrecognized command line argument '";
            public static string UsageStartText = "Usage For '";
            public static string NoInputIni = "No input ini specified. Raw Features will be used.";
            public static string AddingMinMax = "Automatically adding a MinMax normalization transform";
            public static string UseH = "using -h 0 may be faster";
            public static string LbfgsMultiThreading = "LBFGS multi-threading will attempt to load dataset into memory";
            public static string CudaInitFail = "Failed to initialize CUDA runtime";
            public static string NormWarn = "norm=Warn";
            public static string NormNo = "norm=No";
            public static string OneThread = "numThreads=1";
            public static string NoCache = "cache=-";
            public static string NoTrainingInstancesMessage = "*** Exception: 'No valid training instances found, all instances have missing features.";
            public static string NoTrainingInstancesNNMessage = "*** Exception: 'No valid instances found, either because of missing features or the batch size being too large!";
            public static string ConcatDifferentTypes = "All source columns must have the same type.";
            public static string FastTreeDiskTranspose = "FastTree diskTranspose+ (or dt+) option";
            public static string ImageFeaturizerModelFile = "*** Exception: 'Deep neural network model file is missing";
            public static string ImageFeaturizerXmlFile = "*** Exception: 'Deep neural network resource XML file is missing";
            public static string SSWEFeaturizerModelFile = "*** Exception: 'Missing resource for SSWE transform.";
            public static string MlNetImageFeaturizerLink = "https://aka.ms/tlc-image-resources";
        }

        private static string MmlImageFeaturizerLink = "https://go.microsoft.com/fwlink/?linkid=838463";
    }

    /// <summary>
    /// Defines rule to match and decorate message.
    /// </summary>
    class MessageRule
    {
        public Func<string, bool> IsMatch { get; private set; }
        public Func<string, string> Decorate { get; private set; }

        public MessageRule(Func<string, bool> isMatch, Func<string, string> decorate)
        {
            Contracts.CheckValue(isMatch, nameof(isMatch));
            Contracts.CheckValue(decorate, nameof(decorate));

            IsMatch = isMatch;
            Decorate = decorate;
        }
    }

    public static class RmlComponents
    {
        public static readonly string Text = "featurizeText";
        public static readonly string Cat = "categorical";
        public static readonly string CatHash = "categoricalHash";
        public static readonly string Concat = "concat";
        public static readonly string FeatureSelection = "selectFeatures";
        public static readonly string FastTree = "rxFastTrees";
        public static readonly string Sdca = "rxFastLinear";
        public static readonly string FastForest = "rxFastForest";
        public static readonly string LogisticRegression = "rxLogisticRegression";
        public static readonly string NeuralNet = "rxNeuralNet";
        public static readonly string OneClassSvm = "rxOneClassSvm";
    }
}
