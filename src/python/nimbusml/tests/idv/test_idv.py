# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import unittest

import numpy as np
import pandas as pd
from nimbusml import Pipeline, FileDataStream, BinaryDataStream
from nimbusml.datasets import get_dataset
from nimbusml.feature_extraction.categorical import OneHotVectorizer
from nimbusml.linear_model import FastLinearRegressor, OnlineGradientDescentRegressor
from nimbusml.preprocessing.normalization import MinMaxScaler
from nimbusml.preprocessing.schema import ColumnDropper
from sklearn.utils.testing import assert_true, assert_array_equal

# data input (as a FileDataStream)
path = get_dataset('infert').as_filepath()
data = FileDataStream.read_csv(
    path,
    sep=',',
    numeric_dtype=np.float32)  # Error with integer input

def is_nan(x):
    return (x is np.nan or x != x)

def assert_2d_array_equal(actual, desired):
    if len(actual) != len(desired):
        assert_true(False, "arrays are of different lengths.")

    for i in range(len(actual)):
        if len(actual[i]) != len(desired[i]):
            assert_true(False, "arrays are of different lengths.")
        for y in range(len(actual[i])):
            if is_nan(actual[i][y]) and is_nan(desired[i][y]):
                continue
            assert_true(actual[i][y] == desired[i][y])


def transform_data():
    xf = MinMaxScaler(columns={'in': 'induced', 'sp': 'spontaneous'})
    pipe = Pipeline([xf])
    transformed_data = pipe.fit_transform(data, as_binary_data_stream=True)
    transformed_data_df = pipe.fit_transform(data)
    return transformed_data, transformed_data_df


class TestIdv(unittest.TestCase):

    def test_fit_transform(self):
        transformed_data, transformed_data_df = transform_data()
        assert_true(isinstance(transformed_data, BinaryDataStream))
        transformed_data_as_df = transformed_data.to_df()

        assert_true(isinstance(transformed_data_df, pd.DataFrame))
        assert_array_equal(
            transformed_data_as_df.columns,
            transformed_data_df.columns)
        assert_2d_array_equal(
            transformed_data_as_df.values,
            transformed_data_df.values)

    def test_predict(self):
        transformed_data, transformed_data_df = transform_data()
        fl = FastLinearRegressor(
            feature=[
                'parity',
                'in',
                'sp',
                'stratum'],
            label='age')
        flpipe = Pipeline([fl])
        flpipe.fit(transformed_data)
        scores = flpipe.predict(transformed_data)
        scores_df = flpipe.predict(transformed_data_df)

        assert_array_equal(scores, scores_df)

    def test_test(self):
        transformed_data, transformed_data_df = transform_data()
        fl = FastLinearRegressor(
            feature=[
                'parity',
                'in',
                'sp',
                'stratum'],
            label='age')
        flpipe = Pipeline([fl])
        flpipe.fit(transformed_data)
        metrics, scores = flpipe.test(transformed_data, output_scores=True)
        metrics_df, scores_df = flpipe.test(
            transformed_data_df, output_scores=True)

        assert_array_equal(scores, scores_df)
        assert_array_equal(metrics, metrics_df)

        flpipe.fit(
            transformed_data_df.drop(
                'age',
                axis=1),
            transformed_data_df['age'])
        metrics, scores = flpipe.test(transformed_data, output_scores=True)
        metrics_df, scores_df = flpipe.test(
            transformed_data_df, output_scores=True)

        assert_array_equal(scores, scores_df)
        assert_array_equal(metrics, metrics_df)

    def test_fit_predictor_with_idv(self):
        train_data = {'c0': ['a', 'b', 'a', 'b'],
                      'c1': [1, 2, 3, 4],
                      'c2': [2, 3, 4, 5]}
        train_df = pd.DataFrame(train_data).astype({'c1': np.float64,
                                                    'c2': np.float64})

        test_data = {'c0': ['a', 'b', 'b'],
                     'c1': [1.5, 2.3, 3.7],
                     'c2': [2.2, 4.9, 2.7]}
        test_df = pd.DataFrame(test_data).astype({'c1': np.float64,
                                                  'c2': np.float64})

        # Fit a transform pipeline to the training data
        transform_pipeline = Pipeline([OneHotVectorizer() << 'c0'])
        transform_pipeline.fit(train_df)
        df = transform_pipeline.transform(train_df, as_binary_data_stream=True)

        # Fit a predictor pipeline given a transformed BinaryDataStream
        predictor = OnlineGradientDescentRegressor(label='c2', feature=['c0', 'c1'])
        predictor_pipeline = Pipeline([predictor])
        predictor_pipeline.fit(df)

        # Perform a prediction given the test data using
        # the transform and predictor defined previously.
        df = transform_pipeline.transform(test_df, as_binary_data_stream=True)
        result_1 = predictor_pipeline.predict(df)

        # Create expected result
        xf = OneHotVectorizer() << 'c0'
        df = xf.fit_transform(train_df)
        predictor = OnlineGradientDescentRegressor(label='c2', feature=['c0.a', 'c0.b', 'c1'])
        predictor.fit(df)
        df = xf.transform(test_df)
        expected_result = predictor.predict(df)

        self.assertTrue(result_1.loc[:, 'Score'].equals(expected_result))

    def test_fit_transform_with_idv(self):
        path = get_dataset('infert').as_filepath()
        data = FileDataStream.read_csv(path)

        featurization_pipeline = Pipeline([OneHotVectorizer(columns={'education': 'education'})])
        featurization_pipeline.fit(data)
        featurized_data = featurization_pipeline.transform(data, as_binary_data_stream=True)

        schema = featurized_data.schema
        num_columns = len(schema)
        self.assertTrue('case' in schema)
        self.assertTrue('row_num' in schema)

        pipeline = Pipeline([ColumnDropper() << ['case', 'row_num']])
        pipeline.fit(featurized_data)
        result = pipeline.transform(featurized_data)

        columns = result.columns
        self.assertEqual(len(columns), num_columns - 2)
        self.assertTrue('case' not in columns)
        self.assertTrue('row_num' not in columns)


if __name__ == '__main__':
    unittest.main()
