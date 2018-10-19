# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import unittest

import numpy
from nimbusml import Pipeline
from nimbusml.datasets import get_dataset
from nimbusml.decomposition import PcaTransformer
from nimbusml.preprocessing.schema import ColumnConcatenator
from sklearn.utils.testing import assert_almost_equal


class TestPcaTransformer(unittest.TestCase):

    def test_PcaTransformer(self):
        df = get_dataset("infert").as_df()
        X = [
            'age',
            'parity',
            'induced',
            'spontaneous',
            'stratum',
            'pooled.stratum']
        pipe = Pipeline([
            ColumnConcatenator() << {'X': X},
            PcaTransformer(rank=3) << 'X'
        ])
        y = pipe.fit_transform(df[X].astype(numpy.float32))
        y = y[['X.0', 'X.1', 'X.2']]
        assert_almost_equal(
            y.sum().sum(),
            11.293087,
            decimal=3,
            err_msg="Sum should be %s" %
                    11.293087)

    def test_PcaTransformer_no_concat(self):
        df = get_dataset("infert").as_df()
        X = [
            'age',
            'parity',
            'induced',
            'spontaneous',
            'stratum',
            'pooled.stratum']
        pipe = Pipeline([PcaTransformer(rank=3) << [
            'age', 'parity', 'spontaneous', 'stratum']])
        y = pipe.fit_transform(df[X].astype(numpy.float32))
        assert y is not None

    def test_PcaTransformer_int(self):
        df_ = get_dataset("infert").as_df()
        res = {}
        dt = {}
        for ty in (int, float):
            df = df_.copy()
            df['age'] = df['age'].astype(ty)
            df['parity'] = df['parity'].astype(ty)
            df['spontaneous'] = df['spontaneous'].astype(ty)
            df['stratum'] = df['stratum'].astype(ty)
            X = ['age', 'parity', 'spontaneous', 'stratum']
            pipe = Pipeline([
                ColumnConcatenator() << {'X': X},
                PcaTransformer(rank=3) << 'X'
            ])
            y = pipe.fit_transform(df[X], verbose=0)
            res[ty] = y.sum().sum()
            dt[ty] = list(y.dtypes)
        vals = list(res.values())
        assert_almost_equal(vals[0], vals[1])
        dt = list(dt.values())
        dt[0].sort()
        dt[1].sort()
        assert dt[0] != dt[1]


if __name__ == '__main__':
    unittest.main()
