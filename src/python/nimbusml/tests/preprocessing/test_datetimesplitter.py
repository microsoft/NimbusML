# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import platform
import unittest

import pandas
from nimbusml import Pipeline
from nimbusml.preprocessing import DateTimeSplitter
from nimbusml.preprocessing.schema import ColumnSelector
from sklearn.utils.testing import assert_equal


@unittest.skipIf('centos' in platform.linux_distribution()[0].lower(), "centos is not supported")
class TestDateTimeSplitter(unittest.TestCase):

    def test_check_estimator_DateTimeSplitter(self):
        df = pandas.DataFrame(data=dict(dt=[i for i in range(8)]))
        dt = DateTimeSplitter(prefix='dt_') << 'dt'
        result = dt.fit_transform(df)
        assert_equal(result['dt_Year'][0], 1970, "it should have been year of 1970")

    def test_holidays(self):
        df = pandas.DataFrame(data=dict(
            tokens1=[1, 2, 3, 157161600],
            tokens2=[10, 11, 12, 13]
        ))

        cols_to_drop = [
            'dtHour12', 'dtDayOfWeek', 'dtDayOfQuarter',
            'dtDayOfYear', 'dtWeekOfMonth', 'dtQuarterOfYear',
            'dtHalfOfYear', 'dtWeekIso', 'dtYearIso', 'dtMonthLabel',
            'dtAmPmLabel', 'dtDayOfWeekLabel', 'dtIsPaidTimeOff'
        ]

        dts = DateTimeSplitter(prefix='dt', country='Canada') << 'tokens1'
        pipeline = Pipeline([dts, ColumnSelector(drop_columns=cols_to_drop)])
        y = pipeline.fit_transform(df)

        self.assertEqual(y.loc[3, 'dtHolidayName'], 'Christmas Day')

if __name__ == '__main__':
    unittest.main()
