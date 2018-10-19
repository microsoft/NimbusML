# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import unittest

import pandas
from nimbusml.preprocessing.filter import TakeFilter
from sklearn.utils.testing import assert_equal


class TestTakeFilter(unittest.TestCase):

    def test_TakeFilter(self):
        df = pandas.DataFrame(data=dict(
            review=['row' + str(i) for i in range(10)]))
        y = TakeFilter(count=8).fit_transform(df)
        print(len(y))
        assert_equal(len(y), 8)


if __name__ == '__main__':
    unittest.main()
