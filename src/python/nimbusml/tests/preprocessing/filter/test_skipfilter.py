# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import unittest

import pandas
from nimbusml.preprocessing.filter import SkipFilter
from sklearn.utils.testing import assert_equal


class TestSkipFilter(unittest.TestCase):

    def test_SkipFilter(self):
        df = pandas.DataFrame(data=dict(
            review=['row' + str(i) for i in range(10)]))
        y = SkipFilter(count=2).fit_transform(df)
        print(len(y))
        assert_equal(len(y), 8)


if __name__ == '__main__':
    unittest.main()
