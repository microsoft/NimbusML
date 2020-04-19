# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import platform
import unittest

import numpy as np
import pandas as pd
from nimbusml.timeseries import ShortDrop


@unittest.skipIf('centos' in platform.linux_distribution()[0].lower(), "centos is not supported")
class TestShortDrop(unittest.TestCase):

    def test_no_drops(self):

        df = pd.DataFrame(data=dict(
            ts=[1.0, 3.0, 5.0, 7.0],
            grain=['1970', '1970', '1970', '1970'],
        ))

        sd = ShortDrop(grain_columns=['grain'], min_rows=4) << 'ts'
        result = sd.fit_transform(df)
        pd.testing.assert_frame_equal(result, df)

    def test_drop_all(self):

        df = pd.DataFrame(data=dict(
            ts=[1.0, 3.0, 5.0, 7.0],
            grain=['1970', '1970', '1970', '1970'],
        ))

        sd = ShortDrop(grain_columns=['grain'], min_rows=100) << 'ts'
        result = sd.fit_transform(df)
        self.assertEqual(len(result), 0)

if __name__ == '__main__':
    unittest.main()
