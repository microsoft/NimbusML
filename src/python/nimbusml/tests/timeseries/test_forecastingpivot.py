# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import platform
import unittest

import numpy as np
import pandas as pd
from nimbusml import Pipeline
from nimbusml.timeseries import ForecastingPivot, RollingWindow

# BUGS
# Removes NaN values? Record 0 is removed
@unittest.skipIf('centos' in platform.linux_distribution()[0].lower(), "centos is not supported")
class TestForecastingPivot(unittest.TestCase):

    def test_simple_pivot(self):

        df = pd.DataFrame(data=dict(
            ts=[1.0, 3.0, 5.0, 7.0],
            grain=['1970', '1970', '1970', '1970'],
        ))

        rw = RollingWindow(columns={'ts_r': 'ts'},
                           grain_column=['grain'], 
                           window_calculation='Mean',
                           max_window_size=1,
                           horizon=1)

        xf1 = ForecastingPivot(columns_to_pivot=['ts_r'])

        pipe = Pipeline([rw, xf1])

        result = pipe.fit_transform(df)


if __name__ == '__main__':
    unittest.main()
