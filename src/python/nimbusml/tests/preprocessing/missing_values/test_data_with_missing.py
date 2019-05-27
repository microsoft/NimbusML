# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import unittest

import numpy as np
from math import isnan
from nimbusml import Pipeline
from nimbusml.linear_model import FastLinearRegressor
from nimbusml.preprocessing.missing_values import Filter, Handler
from pandas import DataFrame
from sklearn.utils.testing import assert_equal, assert_true, \
    assert_allclose


class TestDataWithMissing(unittest.TestCase):

    def test_missing(self):
        data = DataFrame(data=dict(
            f0=[np.nan, 1, 2, 3, 4, 5, 6],
            f1=[1, 2, np.nan, 3, 4, 5, 6],
            f2=[np.nan, 1, np.nan, 2, 3, np.nan, 4]))

        for col in data.columns:
            xf = Filter(columns=[col])
            filtered = xf.fit_transform(data)
            count = [isinstance(x, str) or not isnan(x)
                     for x in data[col]].count(True)
            assert_equal(filtered.shape[0], count)

    def test_null(self):
        data = DataFrame(data=dict(
            f0=[None, 1, 2, 3, 4, 5, 6],
            f1=[1, 2, np.nan, 3, 4, 5, 6],
            f2=[np.nan, 1, np.nan, 2, 3, np.nan, 4]))

        for col in data.columns:
            xf = Filter(columns=[col])
            filtered = xf.fit_transform(data)
            count = [x is None or isinstance(x, str) or not isnan(
                x) for x in data[col]].count(True)
            count_none = [x is None for x in data[col]].count(True)
            assert_equal(filtered.shape[0], count)
            assert_equal(
                count_none, [
                    x is None for x in filtered[col]].count(True))

    def test_inf(self):
        data = DataFrame(data=dict(
            f0=[np.inf, 1, 2, 3, 4, 5, 6],
            f1=[1, 2, -np.Infinity, 3, 4, 5, 6]))

        xf = Filter(columns=['f0'])
        filtered = xf.fit_transform(data)
        assert_equal(filtered['f0'][0], np.inf)
        assert_equal(filtered['f1'][2], -np.inf)

    def test_input_types(self):
        df = DataFrame(
            data=dict(
                Label=[
                    1, 2, 3, 4, 5], f=[
                    1.1, 2.2, 3.3, np.nan, 5.5], f1=[
                    2.2, np.nan, 4.4, 5.5, 6.6]))
        h = Handler(replace_with='Mean')
        ft = FastLinearRegressor(shuffle=False, number_of_threads=1)
        p = Pipeline([h, ft])
        p.fit(df[['f', 'f1']].values, df['Label'])
        res = p.predict(df[['f', 'f1']].values)
        print(res)
        print(p.summary())
        assert_allclose(
            res['Score'].values, [
                4.965541, 0.519701, 4.992831, 3.877400, 5.020121], rtol=1e-4)


if __name__ == '__main__':
    unittest.main()
