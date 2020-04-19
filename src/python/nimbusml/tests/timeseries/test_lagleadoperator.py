# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import platform
import unittest

import math
import numpy as np
import pandas as pd
from nimbusml.timeseries import LagLeadOperator


@unittest.skipIf('centos' in platform.linux_distribution()[0].lower(), "centos is not supported")
class TestLagLeadOperator(unittest.TestCase):

    def test_no_lag(self):

        df = pd.DataFrame(data=dict(
            ts=[1.0, 3.0, 5.0, 7.0],
            grain=['1970', '1970', '1970', '1970'],
        ))

        ll = LagLeadOperator(columns={'ts_r': 'ts'}, 
                           grain_columns=['grain'], 
                           offsets=[0],
                           horizon=1)

        result = ll.fit_transform(df)
 
        self.assertEqual(result.loc[0, 'ts_r'], 1)
        self.assertEqual(result.loc[1, 'ts_r'], 3)
        self.assertEqual(result.loc[2, 'ts_r'], 5)
        self.assertEqual(result.loc[3, 'ts_r'], 7)

    def test_simple_horizon(self):

        df = pd.DataFrame(data=dict(
            ts=[1.0, 3.0, 5.0, 7.0],
            grain=['1970', '1970', '1970', '1970'],
        ))

        ll = LagLeadOperator(columns={'ts_r': 'ts'}, 
                           grain_columns=['grain'], 
                           offsets=[0],
                           horizon=2)

        result = ll.fit_transform(df)

        self.assertTrue(math.isnan(result.loc[0, 'ts_r.0']))
        self.assertEqual(result.loc[1, 'ts_r.0'], 1)
        self.assertEqual(result.loc[2, 'ts_r.0'], 3)
        self.assertEqual(result.loc[3, 'ts_r.0'], 5)
        
        self.assertEqual(result.loc[0, 'ts_r.1'], 1)
        self.assertEqual(result.loc[1, 'ts_r.1'], 3)
        self.assertEqual(result.loc[2, 'ts_r.1'], 5)
        self.assertEqual(result.loc[3, 'ts_r.1'], 7)

    def test_simple_lag(self):

        df = pd.DataFrame(data=dict(
            ts=[1.0, 3.0, 5.0, 7.0],
            grain=['1970', '1970', '1970', '1970'],
        ))

        ll = LagLeadOperator(columns={'ts_r': 'ts'}, 
                           grain_columns=['grain'], 
                           offsets=[-1, 1],
                           horizon=1)

        result = ll.fit_transform(df)
 
        self.assertTrue(math.isnan(result.loc[0, 'ts_r.0']))
        self.assertEqual(result.loc[1, 'ts_r.0'], 1)
        self.assertEqual(result.loc[2, 'ts_r.0'], 3)
        self.assertEqual(result.loc[3, 'ts_r.0'], 5)

        self.assertEqual(result.loc[0, 'ts_r.1'], 3)
        self.assertEqual(result.loc[1, 'ts_r.1'], 5)
        self.assertEqual(result.loc[2, 'ts_r.1'], 7)
        self.assertTrue(math.isnan(result.loc[3, 'ts_r.1']))

if __name__ == '__main__':
    unittest.main()
