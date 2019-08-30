# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import unittest

import numpy as np
from math import isnan
from nimbusml import Pipeline
from nimbusml.linear_model import FastLinearRegressor
from nimbusml.preprocessing import ToKeyImputer
from nimbusml.preprocessing.missing_values import Filter, Handler, Indicator
from pandas import DataFrame
from sklearn.utils.testing import assert_equal, assert_true, \
    assert_allclose


class TestDataWithMissing(unittest.TestCase):

    def test_category_imputation(self):
        data={'f0': [4, 4, np.nan, 9],
              'f1': [4, 4, np.nan, np.nan]}
        data = DataFrame(data)

        # Check ToKeyImputer
        xf = ToKeyImputer(columns={'f0.out': 'f0', 'f1.out': 'f1'})
        result = xf.fit_transform(data)
        
        assert_equal(result['f0.out'][1], 4)
        assert_equal(result['f0.out'][2], 4)
        assert_equal(result['f1.out'][1], 4)
        assert_equal(result['f1.out'][2], 4)

if __name__ == '__main__':
    unittest.main()
