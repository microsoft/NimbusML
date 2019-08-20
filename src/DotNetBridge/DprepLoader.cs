using Microsoft.ML;
using Microsoft.DataPrep.Common;

namespace Microsoft.MachineLearning.DotNetBridge
{
    internal class DprepLoader1
    {
        public static IDataView Load(string pythonPath, string path)
        {
            DPrepSettings.Instance.PythonPath = pythonPath;
            return DataFlow.FromDPrepFile(path).ToDataView();
        }
    }
}
