# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import os
import time
import tempfile
import unittest
import pandas as pd
import six
from nimbusml import Pipeline, FileDataStream
from nimbusml.datasets import get_dataset
from nimbusml.feature_extraction.text import NGramFeaturizer
from nimbusml.linear_model import AveragedPerceptronBinaryClassifier
from nimbusml.multiclass import OneVsRestClassifier
from nimbusml.preprocessing import DatasetTransformer
from data_frame_tool import DataFrameTool as DFT


def get_tmp_file(suffix=None):
    fd, file_name = tempfile.mkstemp(suffix=suffix)
    fl = os.fdopen(fd, 'w')
    fl.close()
    return file_name

path = get_dataset("wiki_detox_train").as_filepath()
train_set = FileDataStream.read_csv(path, sep='\t')
path = get_dataset("wiki_detox_test").as_filepath()
test_set = FileDataStream.read_csv(path, sep='\t')

class TestOnnxRuntime(unittest.TestCase):
    """
    Tests automl use case:
        1. Fit featurization pipeline separately.
        2. Fit learner on top of the featurization pipeline.
        3. Export to ONNX the learner pipeline.
        4. Compare results between ML.NET and ORT
    """

    @unittest.skipIf(six.PY2, "Disabled due to bug on Mac Python 2.7 build, more info:")
    def test_automl_usecase(self):
        # train featurization pipeline
        featurization_pipe = Pipeline([NGramFeaturizer(keep_diacritics=True, columns={'Features': ['SentimentText']})])
        featurization_pipe.fit(train_set)

        # train learner pipeline
        learner_pipe = Pipeline([DatasetTransformer(featurization_pipe.model),
                    OneVsRestClassifier(AveragedPerceptronBinaryClassifier(),
                                       feature=['Features'], label='Sentiment')
        ])
        learner_pipe.fit(train_set)

        # Export the learner pipeline to ONNX
        onnx_path = get_tmp_file('.onnx')
        learner_pipe.export_to_onnx(onnx_path, 'com.microsoft.ml', onnx_version='Stable')

        # Perform the transform using the standard ML.Net backend
        start = time.time()
        result_standard = learner_pipe.predict(test_set)
        end = time.time()
        print('%ss done transform using standard backend' % round(end -  start, 3))

        # Perform the transform using the ORT backend
        df_tool = DFT(onnx_path)
        dataset = test_set.to_df()
        start = time.time()
        result_ort = df_tool.execute(dataset, ['PredictedLabel.output', 'Score.output'])
        end = time.time()
        print('%ss done transform using ORT backend (excludes df load time)' % round(end - start, 3))

        # compare the results
        for col_tuple in (('PredictedLabel', 'PredictedLabel.output'), 
                          ('Score.0', 'Score.output.0'),
                          ('Score.1', 'Score.output.1'),
                          ):
            col_expected = result_standard.loc[:, col_tuple[0]]
            col_ort = result_ort.loc[:, col_tuple[1]]

            check_kwargs = {
                'check_names': False,
                'check_exact': False,
                'check_dtype': True,
                'check_less_precise': True
            }

            pd.testing.assert_series_equal(col_expected, col_ort, **check_kwargs)

if __name__ == '__main__':
    unittest.main()
