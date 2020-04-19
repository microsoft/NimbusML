# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import platform
import unittest

import math
import numpy as np
import pandas as pd
from nimbusml.timeseries import RollingWindow


# BUGS
# Grain is only string? Fix the error message as in ShortDrop
# Horizon predicitons are not in correct order, see above
@unittest.skipIf('centos' in platform.linux_distribution()[0].lower(), "centos is not supported")
class TestRollingWindow(unittest.TestCase):

    def test_simple_rolling_window(self):

        df = pd.DataFrame(data=dict(
            ts=[1.0, 3.0, 5.0, 7.0],
            grain=['1970', '1970', '1970', '1970'],
        ))

        rw = RollingWindow(columns={'ts_r': 'ts'},
                           grain_column=['grain'], 
                           window_calculation='Mean',
                           max_window_size=1,
                           horizon=2)
        result = rw.fit_transform(df)

        self.assertTrue(math.isnan(result.loc[0, 'ts_r']))
        self.assertEqual(result.loc[1, 'ts_r'], 1)
        self.assertEqual(result.loc[2, 'ts_r'], 3)
        self.assertEqual(result.loc[3, 'ts_r'], 5)

    def test_simple_rolling_window2(self):

        df = pd.DataFrame(data=dict(
            ts=[1.0, 3.0, 5.0, 7.0],
            grain=['1970', '1970', '1970', '1970'],
        ))

        rw = RollingWindow(columns={'ts_r': 'ts'},
                           grain_column=['grain'], 
                           window_calculation='Mean',
                           max_window_size=2,
                           horizon=2)
        result = rw.fit_transform(df)

        self.assertTrue(math.isnan(result.loc[0, 'ts_r.0']))
        self.assertTrue(math.isnan(result.loc[1, 'ts_r.0']))
        self.assertEqual(result.loc[2, 'ts_r.0'], 1)
        self.assertEqual(result.loc[3, 'ts_r.0'], 2)
        
        self.assertTrue(math.isnan(result.loc[0, 'ts_r.1']))
        self.assertEqual(result.loc[1, 'ts_r.1'], 1)
        self.assertEqual(result.loc[2, 'ts_r.1'], 2)
        self.assertEqual(result.loc[3, 'ts_r.1'], 4)

if __name__ == '__main__':
    unittest.main()
