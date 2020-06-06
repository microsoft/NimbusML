# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import platform
import unittest

import numpy as np
from pandas import DataFrame
from nimbusml.preprocessing import ToString
from sklearn.utils.testing import assert_equal


@unittest.skipIf('centos' in platform.linux_distribution()[0].lower(), "centos is not supported")
class TestToString(unittest.TestCase):

    def test_tostring(self):
        data={'f0': [4, 4, -1, 9],
              'f1': [5, 5, 3.1, -0.23],
              'f2': [6, 6.7, np.nan, np.nan]}
        data = DataFrame(data).astype({'f0': np.int32,
                                       'f1': np.float32,
                                       'f2': np.float64})

        xf = ToString(columns={'f0.out': 'f0',
                               'f1.out': 'f1',
                               'f2.out': 'f2'})
        result = xf.fit_transform(data)

        assert_equal(result['f0.out'][1], '4')
        assert_equal(result['f0.out'][2], '-1')
        assert_equal(result['f1.out'][1], '5.000000')
        assert_equal(result['f1.out'][2], '3.100000')
        assert_equal(result['f2.out'][1], '6.700000')
        assert_equal(result['f2.out'][2], 'NaN')

if __name__ == '__main__':
    unittest.main()
