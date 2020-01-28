# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
import platform
import unittest

import pandas
from nimbusml import Pipeline
from nimbusml.preprocessing.normalization import RobustScaler


@unittest.skipIf('centos' in platform.linux_distribution()[0].lower(), "centos is not supported")
class TestRobustScaler(unittest.TestCase):

    def test_with_integer_inputs(self):
        df = pandas.DataFrame(data=dict(c0=[1, 3, 5, 7, 9]))

        xf = RobustScaler(columns='c0', center=True, scale=True)
        pipeline = Pipeline([xf])
        result = pipeline.fit_transform(df)

        expected_result = pandas.Series([-1.0, -0.5, 0.0, 0.5, 1.0])

        self.assertTrue(result.loc[:, 'c0'].equals(expected_result))


if __name__ == '__main__':
    unittest.main()
