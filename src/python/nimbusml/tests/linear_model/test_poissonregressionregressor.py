# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import unittest

import numpy as np
from nimbusml.datasets import get_dataset
from nimbusml.linear_model import PoissonRegressionRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.utils.testing import assert_almost_equal


class TestPoissonRegressionRegressor(unittest.TestCase):

    def test_PoissonRegressionRegressor(self):
        np.random.seed(0)
        df = get_dataset("airquality").as_df().fillna(0)
        df = df[df.Ozone.notnull()]
        X_train, X_test, y_train, y_test = train_test_split(
            df.loc[:, df.columns != 'Ozone'], df['Ozone'])
        poi = PoissonRegressionRegressor().fit(X_train, y_train)
        scores = poi.predict(X_test)
        r2 = r2_score(y_test, scores)
        assert_almost_equal(
            r2,
            0.529520655,
            decimal=3,
            err_msg="Sum should be %s" %
                    0.529520655)


if __name__ == '__main__':
    unittest.main()
