# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import unittest

import numpy as np
import pandas as pd
from nimbusml import Pipeline, FileDataStream, BinaryDataStream
from nimbusml.datasets import get_dataset
from nimbusml.linear_model import FastLinearRegressor
from nimbusml.preprocessing.normalization import MinMaxScaler
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


if __name__ == '__main__':
    unittest.main()
