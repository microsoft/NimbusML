# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import unittest

import pandas as pd
from nimbusml import Pipeline, FileDataStream
from nimbusml.datasets import get_dataset
from nimbusml.timeseries import IidChangePointDetector


class TestIidChangePointDetector(unittest.TestCase):

    def test_correct_data_is_marked_as_change_point(self):
        input_data = [5, 5, 5, 5, 5, 5, 5, 5]
        input_data.extend([7, 7, 7, 7, 7, 7, 7, 7])
        X_train = pd.Series(input_data, name="ts")

        cpd = IidChangePointDetector(confidence=95, change_history_length=4) << {'result': 'ts'}
        data = cpd.fit_transform(X_train)

        self.assertEqual(data.loc[8, 'result.Alert'], 1.0)

        data = data.loc[data['result.Alert'] == 1.0]
        self.assertEqual(len(data), 1)

    def test_multiple_user_specified_columns_is_not_allowed(self):
        path = get_dataset('timeseries').as_filepath()
        data = FileDataStream.read_csv(path)

        try:
            pipeline = Pipeline([
                IidChangePointDetector(columns=['t2', 't3'], change_history_length=5)
            ])
            pipeline.fit_transform(data)

        except RuntimeError as e:
            self.assertTrue('Only one column is allowed' in str(e))
            return

        self.fail()


if __name__ == '__main__':
    unittest.main()
