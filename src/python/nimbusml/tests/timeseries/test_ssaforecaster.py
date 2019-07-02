# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import unittest

import numpy as np
import pandas as pd
from nimbusml import Pipeline, FileDataStream
from nimbusml.datasets import get_dataset
from nimbusml.timeseries import SsaForecaster


class TestSsaForecaster(unittest.TestCase):

    @unittest.skip('ml.net libraries containing timeseries forecasting are not included with nimbusml yet.')
    def test_simple_forecast(self):
        seasonality_size = 5
        seasonal_data = np.arange(seasonality_size)

        data = np.tile(seasonal_data, 3)

        X_train = pd.Series(data, name="ts")

        training_seasons = 3
        training_size = seasonality_size * training_seasons

<<<<<<< HEAD
        forecaster = SsaForecaster(series_length=8,
=======
        forecaster = SsaForecaster(forcasting_confident_lower_bound_column_name="cmin",
                                   forcasting_confident_upper_bound_column_name="cmax",
                                   series_length=8,
>>>>>>> Fixed estimator checks failed from ssa_forecasting's update (#166)
                                   train_size=training_size,
                                   window_size=seasonality_size + 1,
                                   horizon=2) <<   {'fc': 'ts'}

        forecaster.fit(X_train, verbose=1)
        data = forecaster.transform(X_train)

        self.assertEqual(round(data.loc[0, 'fc.0']), 1.0)
        self.assertEqual(round(data.loc[0, 'fc.1']), 2.0)

        self.assertEqual(len(data['fc.0']), 15)

    @unittest.skip('ml.net libraries containing timeseries forecasting are not included with nimbusml yet.')
    def test_multiple_user_specified_columns_is_not_allowed(self):
        path = get_dataset('timeseries').as_filepath()
        data = FileDataStream.read_csv(path)

        try:
            pipeline = Pipeline([
<<<<<<< HEAD
                SsaForecaster(series_length=8,
=======
                SsaForecaster(forcasting_confident_lower_bound_column_name="cmin",
                              forcasting_confident_upper_bound_column_name="cmax",
                              series_length=8,
>>>>>>> Fixed estimator checks failed from ssa_forecasting's update (#166)
                              train_size=15,
                              window_size=5,
                              horizon=2,
                              columns=['t2', 't3'])
            ])
            pipeline.fit_transform(data)

        except RuntimeError as e:
            self.assertTrue('Only one column is allowed' in str(e))
            return

        self.fail()


if __name__ == '__main__':
    unittest.main()
