# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import unittest

import pandas as pd
from nimbusml.preprocessing.filter import RangeFilter


class TestRangeFilter(unittest.TestCase):

    def test_reange_filter_short_example(self):
        d = pd.DataFrame([[1., 1.9, 3.], [2., 3., 4.], [2., 3., 4.]])
        d.columns = ['aa', 'bb', 'cc']

        hdl = RangeFilter(min=0, max=2) << 'aa'
        res1 = hdl.fit_transform(d)
        assert res1 is not None
        assert res1.shape == (1, 3)

        hdl = RangeFilter(min=0, max=2) << 'bb'
        res2 = hdl.fit_transform(d)
        assert res2 is not None
        assert res2.shape == (1, 3)
        assert res1.values.ravel().tolist() == res2.values.ravel().tolist()


if __name__ == '__main__':
    unittest.main()
