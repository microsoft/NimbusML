# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import unittest

import numpy as np
import pandas as pd
from nimbusml import Pipeline, FileDataStream
from nimbusml.datasets import get_dataset
from nimbusml.time_series import IidSpikeDetector
from nimbusml.preprocessing.schema import TypeConverter


class TestIidSpikeDetector(unittest.TestCase):

    def test_correct_data_is_marked_as_anomaly(self):
        X_train = pd.Series([5, 5, 5, 5, 5, 10, 5, 5, 5, 5, 5], name="ts", dtype=np.float32)
        isd = IidSpikeDetector(confidence=95, pvalue_history_length=3) << {'result': 'ts'}
        data = isd.fit_transform(X_train)

        data = data.loc[data['result.Alert'] == 1.0]
        self.assertEqual(len(data), 1)
        self.assertEqual(data.iloc[0]['ts'], 10.0)

    def test_multiple_user_specified_columns_is_not_allowed(self):
        path = get_dataset('timeseries').as_filepath()
        data = FileDataStream.read_csv(path)

        try:
            pipeline = Pipeline([
                TypeConverter(result_type='R4'),
                IidSpikeDetector(columns=['t2', 't3'], pvalue_history_length=5)
            ])
            pipeline.fit_transform(data)

        except RuntimeError as e:
            self.assertTrue('Only one column is allowed' in str(e))
            return

        self.fail()


if __name__ == '__main__':
    unittest.main()
