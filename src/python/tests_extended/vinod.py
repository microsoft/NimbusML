import os
import tempfile
import nimbusml.linear_model as nml_linear
from nimbusml.feature_extraction.categorical import OneHotVectorizer
from nimbusml.preprocessing.missing_values import Handler
from nimbusml import FileDataStream
from nimbusml.preprocessing import DatasetTransformer
from nimbusml import Pipeline
from nimbusml.preprocessing import OnnxRunner

def get_tmp_file(suffix=None):
    fd, file_name = tempfile.mkstemp(suffix=suffix)
    fl = os.fdopen(fd, 'w')
    fl.close()
    return file_name

X_train_dprep = FileDataStream.read_csv("E:/sources/vinod/NYCTaxiTipPrediction_train.csv")
X_test_dprep = FileDataStream.read_csv("E:/sources/vinod/NYCTaxiTipPrediction_valid.csv")

try:
    pipe_featurization = Pipeline([OneHotVectorizer(columns={'vendor_id': 'vendor_id', 'payment_type': 'payment_type', 'passenger_count': 'passenger_count','rate_code': 'rate_code'})
                                      ,Handler(columns={'trip_distance': 'trip_distance', 'trip_time_in_secs': 'trip_time_in_secs'})
    ])
    pipe_featurization.fit(X_train_dprep)

    pipe_training = Pipeline([DatasetTransformer(pipe_featurization.model),
                nml_linear.FastLinearRegressor(feature=['vendor_id', 'payment_type', 'passenger_count', 'rate_code', 'trip_distance', 'trip_time_in_secs'],label='fare_amount')
    ])
    pipe_training.fit(X_train_dprep)

    metrics, scores = pipe_training.test(X_test_dprep)
    print(metrics)
    print('training done')

    # Export the pipeline to ONNX
    onnx_path = get_tmp_file('.onnx')
    pipe_training.export_to_onnx(onnx_path, 'com.microsoft.ml', onnx_version='Stable')
    print('export done')

    # Perform the transform using the standard ML.Net backend
    result_standard = pipe_training.transform(X_test_dprep)
    print(result_standard)
    print('done transform using standard backend')
    #          c1        c2
    # 0  0.025025  0.000998
    # 1  0.305305  0.000998

    # Perform the transform using the ONNX backend.
    # Note, the extra columns and column name differences
    # is a known issue with the ML.Net backend.
    onnxrunner = OnnxRunner(model_file=onnx_path)
    result_onnx = onnxrunner.fit_transform(X_test_dprep)
    print('done transform using onnx backend')
    print(result_onnx)
    #      c1   c2     c12.0     c22.0
    # 0   2.5  1.0  0.025025  0.000998
    # 1  30.5  1.0  0.305305  0.000998

except Exception as e:
    print('tragedy')
    print(e)

print ("done")