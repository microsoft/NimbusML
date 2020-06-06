# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import unittest

import numpy as np
import pandas as pd
from nimbusml import Pipeline
from nimbusml.preprocessing.normalization import LpScaler
from nimbusml.preprocessing.schema import ColumnConcatenator
from sklearn.utils.testing import assert_greater, assert_less


class TestLpScaler(unittest.TestCase):

    def test_lpscaler(self):
        in_df = pd.DataFrame(
            data=dict(
                Sepal_Length=[2.5, 1, 2.1, 1.0],
                Sepal_Width=[.75, .9, .8, .76],
                Petal_Length=[0, 2.5, 2.6, 2.4],
                Species=["setosa", "viginica", "setosa", 'versicolor']))

        in_df.iloc[:, 0:3] = in_df.iloc[:, 0:3].astype(np.float32)

        src_cols = ['Sepal_Length', 'Sepal_Width', 'Petal_Length']

        pipeline = Pipeline([
            ColumnConcatenator() << {'concat': src_cols},
            LpScaler() << {'norm': 'concat'}
        ])
        out_df = pipeline.fit_transform(in_df)

        cols = ['concat.' + s for s in src_cols]
        cols.extend(['norm.' + s for s in src_cols])
        sum = out_df[cols].sum().sum()
        sum_range = (23.24, 23.25)
        assert_greater(sum, sum_range[0], "sum should be greater than %s" % sum_range[0])
        assert_less(sum, sum_range[1], "sum should be less than %s" % sum_range[1])

    def test_lpscaler_automatically_converts_to_single(self):
        in_df = pd.DataFrame(
            data=dict(
                Sepal_Length=[2.5, 1, 2.1, 1.0],
                Sepal_Width=[.75, .9, .8, .76],
                Petal_Length=[0, 2.5, 2.6, 2.4],
                Species=["setosa", "viginica", "setosa", 'versicolor']))

        in_df.iloc[:, 0:3] = in_df.iloc[:, 0:3].astype(np.float64)

        src_cols = ['Sepal_Length', 'Sepal_Width', 'Petal_Length']

        pipeline = Pipeline([
            ColumnConcatenator() << {'concat': src_cols},
            LpScaler() << {'norm': 'concat'}
        ])
        out_df = pipeline.fit_transform(in_df)

        cols = ['concat.' + s for s in src_cols]
        cols.extend(['norm.' + s for s in src_cols])
        sum = out_df[cols].sum().sum()
        sum_range = (23.24, 23.25)
        assert_greater(sum, sum_range[0], "sum should be greater than %s" % sum_range[0])
        assert_less(sum, sum_range[1], "sum should be less than %s" % sum_range[1])


if __name__ == '__main__':
    unittest.main()
