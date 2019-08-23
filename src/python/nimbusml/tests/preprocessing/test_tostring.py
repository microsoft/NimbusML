# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import unittest

import numpy as np
from pandas import DataFrame
from nimbusml.preprocessing import ToString


class TestFromKey(unittest.TestCase):

    def test_tostring(self):
        data={'f0': [4, 4, -1, 9],
              'f1': [4, 4, np.nan, np.nan]}
        data = DataFrame(data)

        xf = ToString(columns={'f0.out': 'f0'})
        result = xf.fit_transform(data)
        assert_equal(result.loc[2, 'f0.out'], "4")
        assert_equal(len(result), 3)

if __name__ == '__main__':
    unittest.main()
