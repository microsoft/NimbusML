# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import platform
import unittest

import numpy as np
import pandas as pd
from nimbusml.timeseries import TimeSeriesImputer


@unittest.skipIf('centos' in platform.linux_distribution()[0].lower(), "centos is not supported")
class TestTimeSeriesImputer(unittest.TestCase):

    def test_timeseriesimputer_adds_new_row(self):
        from nimbusml.timeseries import TimeSeriesImputer

        df = pd.DataFrame(data=dict(
            ts=[1, 2, 3, 5],
            grain=[1970, 1970, 1970, 1970],
            c3=[10, 13, 15, 20],
            c4=[19, 12, 16, 19]
        ))

        tsi = TimeSeriesImputer(time_series_column='ts',
                                grain_columns=['grain'],
                                filter_columns=['c3', 'c4'],
                                impute_mode='ForwardFill',
                                filter_mode='Include')
        result = tsi.fit_transform(df)

        self.assertEqual(result.loc[0, 'ts'], 1)
        self.assertEqual(result.loc[3, 'ts'], 4)
        self.assertEqual(result.loc[3, 'grain'], 1970)
        self.assertEqual(result.loc[3, 'c3'], 15)
        self.assertEqual(result.loc[3, 'c4'], 16)
        self.assertEqual(result.loc[3, 'IsRowImputed'], True)


if __name__ == '__main__':
    unittest.main()
