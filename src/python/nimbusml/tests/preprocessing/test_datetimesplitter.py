# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import unittest

import pandas
from nimbusml.preprocessing import DateTimeSplitter
from sklearn.utils.testing import assert_equal


class TestDateTimeSplitter(unittest.TestCase):

    def test_check_estimator_DateTimeSplitter(self):
        df = pandas.DataFrame(data=dict(dt=[i for i in range(8)]))
        dt = DateTimeSplitter(prefix='dt_') << 'dt'
        result = dt.fit_transform(df)
        assert_equal(0, 1, "to be fixed")


if __name__ == '__main__':
    unittest.main()
