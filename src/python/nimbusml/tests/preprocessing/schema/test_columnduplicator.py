# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import unittest

import pandas
from nimbusml.preprocessing.schema import ColumnDuplicator
from sklearn.utils.testing import assert_equal


class TestColumnDuplicator(unittest.TestCase):

    def test_check_estimator_ColumnDuplicator(self):
        df = pandas.DataFrame(data=dict(
            tokens1=['one_' + str(i) for i in range(8)],
            tokens2=['two_' + str(i) for i in range(8)]
        ))
        cd = ColumnDuplicator() << {'tokens3': 'tokens1'}
        y = cd.fit_transform(df)
        sum = 0
        for v in y.values:
            for c in str(v):
                sum = sum + ord(c)
        assert_equal(sum, 15292, "sum of chars should be %s" % 15292)

    def test_check_estimator_ColumnDuplicator_pairs(self):
        df = pandas.DataFrame(data=dict(
            tokens1=['one_' + str(i) for i in range(8)],
            tokens2=['two_' + str(i) for i in range(8)]
        ))
        cd = ColumnDuplicator() << {'tokens3': 'tokens1', 'tokens4': 'tokens1'}
        y = cd.fit_transform(df)
        sum = 0
        for v in y.values:
            for c in str(v):
                sum = sum + ord(c)
        assert_equal(sum, 19920, "sum of chars should be %s" % 19920)


if __name__ == '__main__':
    unittest.main()
