# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import unittest

import numpy as np
import pandas as pd
from nimbusml import Pipeline, FileDataStream
from nimbusml.datasets import get_dataset
from nimbusml.timeseries import SsaSpikeDetector


class TestSsaSpikeDetector(unittest.TestCase):

    def test_correct_data_is_marked_as_spike(self):
        seasonality_size = 5
        seasonal_data = np.arange(seasonality_size)

        data = np.tile(seasonal_data, 3)
        data = np.append(data, [100]) # add a spike
        data = np.append(data, seasonal_data)

        X_train = pd.Series(data, name="ts")
        training_seasons = 3
        training_size = seasonality_size * training_seasons

        ssd = SsaSpikeDetector(confidence=95,
                               pvalue_history_length=8,
                               training_window_size=training_size,
                               seasonal_window_size=seasonality_size + 1) << {'result': 'ts'}

        ssd.fit(X_train)
        data = ssd.transform(X_train)

        self.assertEqual(data.loc[15, 'result.Alert'], 1.0)

        data = data.loc[data['result.Alert'] == 1.0]
        self.assertEqual(len(data), 1)

    def test_multiple_user_specified_columns_is_not_allowed(self):
        path = get_dataset('timeseries').as_filepath()
        data = FileDataStream.read_csv(path)

        try:
            pipeline = Pipeline([
                SsaSpikeDetector(columns=['t2', 't3'], pvalue_history_length=5)
            ])
            pipeline.fit_transform(data)

        except RuntimeError as e:
            self.assertTrue('Only one column is allowed' in str(e))
            return

        self.fail()


if __name__ == '__main__':
    unittest.main()
