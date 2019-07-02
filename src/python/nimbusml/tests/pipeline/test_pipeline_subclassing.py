# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
import os
import unittest

import numpy as np
import pandas as pd
from nimbusml import Pipeline
from nimbusml.linear_model import LogisticRegressionBinaryClassifier


def generate_dataset_1():
    X = pd.DataFrame({'x1': [2, 3, 2, 2, 8, 9, 10, 8],
                      'x2': [1, 2, 3, 1, 7, 10, 9, 8]})
    y = pd.DataFrame({'y': [1, 1, 1, 1, 0, 0, 0, 0]})
    return X, y


class CustomPipeline(Pipeline):
    # Override the predict method
    def predict(self, X, *args, **kwargs):
        return kwargs.get('test_return_value')


class TestPipelineSubclassing(unittest.TestCase):

    def test_pipeline_subclass_can_override_predict(self):
        X, y = generate_dataset_1()

        pipeline = Pipeline([LogisticRegressionBinaryClassifier()])
        pipeline.fit(X, y)
        result = pipeline.predict(X)['PredictedLabel']

        self.assertTrue(np.array_equal(result.values, y['y'].values))

        pipeline = CustomPipeline([LogisticRegressionBinaryClassifier()])
        pipeline.fit(X, y)

        self.assertEqual(pipeline.predict(X, test_return_value=3), 3)


    def test_pipeline_subclass_correctly_supports_predict_proba(self):
        X, y = generate_dataset_1()

        pipeline = Pipeline([LogisticRegressionBinaryClassifier()])
        pipeline.fit(X, y)
        orig_result = pipeline.predict_proba(X)

        pipeline = CustomPipeline([LogisticRegressionBinaryClassifier()])
        pipeline.fit(X, y)
        new_result = pipeline.predict_proba(X)

        self.assertTrue(np.array_equal(orig_result, new_result))


    def test_pipeline_subclass_correctly_supports_decision_function(self):
        X, y = generate_dataset_1()

        pipeline = Pipeline([LogisticRegressionBinaryClassifier()])
        pipeline.fit(X, y)
        orig_result = pipeline.decision_function(X)

        pipeline = CustomPipeline([LogisticRegressionBinaryClassifier()])
        pipeline.fit(X, y)
        new_result = pipeline.decision_function(X)

        self.assertTrue(np.array_equal(orig_result, new_result))


if __name__ == '__main__':
    unittest.main()
