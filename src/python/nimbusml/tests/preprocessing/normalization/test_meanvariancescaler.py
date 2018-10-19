# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
import unittest
from collections import OrderedDict

import pandas
from nimbusml import Pipeline
from nimbusml.preprocessing.normalization import MeanVarianceScaler
from sklearn.utils.testing import assert_almost_equal, assert_equal


class TestMeanVarianceScaler(unittest.TestCase):

    def test_transform_float(self):
        in_df = pandas.DataFrame(data=dict(xpetal=[-1.1, -2.2, -3.3],
                                           ipetal=[1, 2, 3]))

        normed = MeanVarianceScaler() << ['xpetal', 'ipetal']
        pipeline = Pipeline([normed])
        out_df = pipeline.fit_transform(in_df, verbose=0)
        assert_equal(out_df.shape, (3, 2))
        assert_almost_equal(out_df.loc[2, 'xpetal'], -1.3887302, decimal=3)
        assert_almost_equal(out_df.loc[2, 'ipetal'], 1.38873, decimal=3)

    def test_transform_int(self):
        in_df = pandas.DataFrame(data=dict(xpetal=[-1, -2, -3],
                                           ipetal=[1, 2, 3]))

        normed = MeanVarianceScaler() << ['xpetal', 'ipetal']
        pipeline = Pipeline([normed])
        out_df = pipeline.fit_transform(in_df, verbose=0)
        assert_equal(out_df.shape, (3, 2))
        assert_almost_equal(out_df.loc[2, 'xpetal'], -1.3887302, decimal=3)
        assert_almost_equal(out_df.loc[2, 'ipetal'], 1.38873, decimal=3)

    def test_transform_int_rename(self):
        in_df = pandas.DataFrame(data=dict(xpetal=[-1, -2, -3],
                                           ipetal=[1, 2, 3]))

        normed = MeanVarianceScaler() << OrderedDict(
            [('ii', 'xpetal'), ('jj', 'ipetal')])
        pipeline = Pipeline([normed])
        out_df = pipeline.fit_transform(in_df, verbose=0)
        assert_equal(out_df.shape, (3, 4))
        assert_almost_equal(out_df.loc[2, 'ii'], -1.3887302, decimal=3)
        assert_almost_equal(out_df.loc[2, 'jj'], 1.38873, decimal=3)


if __name__ == '__main__':
    unittest.main()
