import os
import tempfile
import numpy as np
import pandas as pd
from nimbusml import Pipeline
from nimbusml.preprocessing import OnnxRunner
from nimbusml.preprocessing.normalization import MinMaxScaler


def get_tmp_file(suffix=None):
    fd, file_name = tempfile.mkstemp(suffix=suffix)
    fl = os.fdopen(fd, 'w')
    fl.close()
    return file_name

# Generate the train and test data
np.random.seed(0)
x = np.arange(100, step=0.1)
y = x * 10 + (np.random.standard_normal(len(x)) * 10)
train_data = {'c1': x, 'c2': y}
train_df = pd.DataFrame(train_data).astype({'c1': np.float32, 'c2': np.float32})

test_data = {'c1': [2.5, 30.5], 'c2': [1, 1]}
test_df = pd.DataFrame(test_data).astype({'c1': np.float32, 'c2': np.float32})

# Fit a MinMaxScaler Pipeline
r1 = Pipeline([MinMaxScaler()])
r1.fit(train_df)

# Export the pipeline to ONNX
onnx_path = get_tmp_file('.onnx')
r1.export_to_onnx(onnx_path, 'com.microsoft.ml', onnx_version='Stable')

# Perform the transform using the standard ML.Net backend
result_standard = r1.transform(test_df)
print(result_standard)
#          c1        c2
# 0  0.025025  0.000998
# 1  0.305305  0.000998

# Perform the transform using the ONNX backend.
# Note, the extra columns and column name differences
# is a known issue with the ML.Net backend.
onnxrunner = OnnxRunner(model_file=onnx_path)
result_onnx = onnxrunner.fit_transform(test_df)
print(result_onnx)
#      c1   c2     c12.0     c22.0
# 0   2.5  1.0  0.025025  0.000998
# 1  30.5  1.0  0.305305  0.000998
