# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import unittest

import numpy as np
from pandas import DataFrame
from nimbusml.preprocessing import ToString
from sklearn.utils.testing import assert_equal


class TestToString(unittest.TestCase):

    def test_tostring(self):
        data={'f0': [4, 4, -1, 9],
              'f1': [4, 4, np.nan, np.nan]}
        data = DataFrame(data)

        xf = ToString(columns={'f0.out': 'f0', 'f1.out': 'f1'})
        result = xf.fit_transform(data)
        
        assert_equal(result['f0.out'][1], '4')
        assert_equal(result['f0.out'][2], '-1')
        assert_equal(result['f1.out'][1], '4')
        assert_equal(result['f1.out'][2], 'NaN')

if __name__ == '__main__':
    unittest.main()
