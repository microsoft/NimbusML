# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import unittest
from collections import OrderedDict

import pandas
from nimbusml import Pipeline
from nimbusml.preprocessing.normalization import MinMaxScaler
from sklearn.utils.testing import assert_equal


class TestMinMaxScaler(unittest.TestCase):

    def test_minmaxscaler_float(self):
        in_df = pandas.DataFrame(data=dict(xpetal=[-1.1, -2.2, -3.3],
                                           ipetal=[1, 2, 3]))

        normed = MinMaxScaler() << ['xpetal', 'ipetal']
        pipeline = Pipeline([normed])
        out_df = pipeline.fit_transform(in_df, verbose=0)
        assert_equal(out_df.shape, (3, 2))
        if out_df.loc[2, 'xpetal'] != -1:
            raise Exception("Unexpected:\n" + str(out_df))
        assert_equal(out_df.loc[2, 'ipetal'], 1)

    def test_minmaxscaler_int(self):
        in_df = pandas.DataFrame(data=dict(xpetal=[-1, -2, -3],
                                           ipetal=[1, 2, 3]))

        normed = MinMaxScaler() << ['xpetal', 'ipetal']
        pipeline = Pipeline([normed])
        out_df = pipeline.fit_transform(in_df, verbose=0)
        assert_equal(out_df.shape, (3, 2))
        if out_df.loc[2, 'xpetal'] != -1:
            raise Exception("Unexpected:\n" + str(out_df))
        assert_equal(out_df.loc[2, 'ipetal'], 1)

    def test_minmaxscaler_int_rename(self):
        in_df = pandas.DataFrame(data=dict(xpetal=[-1, -2, -3],
                                           ipetal=[1, 2, 3]))

        normed = MinMaxScaler() << OrderedDict(
            [('ii', 'xpetal'), ('jj', 'ipetal')])
        pipeline = Pipeline([normed])
        out_df = pipeline.fit_transform(in_df, verbose=0)
        assert_equal(out_df.shape, (3, 4))
        if out_df.loc[2, 'ii'] != -1:
            raise Exception("Unexpected:\n" + str(out_df))
        assert_equal(out_df.loc[2, 'jj'], 1)

    @unittest.skip('nimbusml does not preserce column ordering.')
    def test_minmaxscaler_float_order_int(self):
        in_df = pandas.DataFrame(data=OrderedDict(xpetal=[-1.1, -2.2, -3.3],
                                                  ipetal=[1, 2, 3]))

        normed = MinMaxScaler() << ['xpetal', 'ipetal']
        pipeline = Pipeline([normed])
        out_df = pipeline.fit_transform(in_df, verbose=0)
        assert_equal(out_df.shape, (3, 2))
        assert_equal(list(out_df.columns), list(in_df.columns))

    @unittest.skip('nimbusml does not preserce column ordering.')
    def test_minmaxscaler_float_order_noint(self):
        in_df = pandas.DataFrame(data=OrderedDict(xpetal=[-1.1, -2.2, -3.3],
                                                  ipetal=[1.0, 2.0, 3.0]))

        normed = MinMaxScaler() << ['xpetal', 'ipetal']
        pipeline = Pipeline([normed])
        out_df = pipeline.fit_transform(in_df, verbose=0)
        assert_equal(out_df.shape, (3, 2))
        assert_equal(list(out_df.columns), list(in_df.columns))


if __name__ == '__main__':
    unittest.main()
