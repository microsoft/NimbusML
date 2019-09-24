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
        assert_equal(result['dt_Year'][0], 1970, "it should have been year of 1970")

    def test_holidays(self):
        df = pandas.DataFrame(data=dict(
            tokens1=[1, 2, 3, 157161600],
            tokens2=[10, 11, 12, 13]
        ))

        cols_to_drop = [
            'Hour12', 'DayOfWeek', 'DayOfQuarter',
            'DayOfYear', 'WeekOfMonth', 'QuarterOfYear',
            'HalfOfYear', 'WeekIso', 'YearIso', 'MonthLabel',
            'AmPmLabel', 'DayOfWeekLabel', 'IsPaidTimeOff'
        ]

        dts = DateTimeSplitter(prefix='dt',
                               country='Canada',
                               columns_to_drop=cols_to_drop) << 'tokens1'
        y = dts.fit_transform(df)

        self.assertEqual(y.loc[3, 'dtHolidayName'], 'Christmas Day')

if __name__ == '__main__':
    unittest.main()
