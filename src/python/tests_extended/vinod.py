# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import os
import time
import tempfile
import nimbusml.linear_model as nml_linear
from nimbusml.feature_extraction.categorical import OneHotVectorizer
from nimbusml.preprocessing.missing_values import Handler
from nimbusml import FileDataStream
from nimbusml.preprocessing import DatasetTransformer
from nimbusml.preprocessing.schema import ColumnSelector
from nimbusml import Pipeline
from nimbusml.preprocessing import OnnxRunner
from data_frame_tool import DataFrameTool as DFT

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
    start = time.time()
    result_standard = pipe_training.predict(X_test_dprep)
    end = time.time()
    print(result_standard)
    print('%ss done transform using standard backend' % round(end - start, 3))

    # Perform the transform using the ONNX backend.
    # Note, the extra columns and column name differences
    # is a known issue with the ML.Net backend.
    onnxrunner = Pipeline([OnnxRunner(model_file=onnx_path), 
                           ColumnSelector(columns=['Score'])])
    # Performance issue, commenting out for now
    #start = time.time()
    #result_onnx = onnxrunner.fit_transform(X_test_dprep, as_binary_data_stream=True)
    #end = time.time()
    #print(result_onnx.head(5))
    #print('%ss done transform using onnx backend' % round(end - start, 3))

    df_tool = DFT(onnx_path)
    dataset = X_test_dprep.to_df()
    start = time.time()
    result_ort = df_tool.execute(dataset, [])
    end = time.time()
    print(result_ort)
    print('%ss done transform using ORT backend (excludes df load time)' % round(end - start, 3))


except Exception as e:
    print('=============== ERROR =================')
    print(e)

print ("done")