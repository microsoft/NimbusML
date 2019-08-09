# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
import platform
import unittest

import numpy as np
from nimbusml.datasets import get_dataset
from nimbusml.ensemble import LightGbmRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.utils.testing import assert_greater, assert_less


class TestLightGbmRegressor(unittest.TestCase):

    def test_lightgbmregressor(self):
        np.random.seed(0)

        df = get_dataset("airquality").as_df().fillna(0)
        df = df[df.Ozone.notnull()]

        X_train, X_test, y_train, y_test = train_test_split(
            df.loc[:, df.columns != 'Ozone'], df['Ozone'])

        # Train a model and score
        ftree = LightGbmRegressor().fit(X_train, y_train)
        scores = ftree.predict(X_test)

        r2 = r2_score(y_test, scores)
        assert_greater(r2, 0.32, "should be greater than %s" % 0.32)
        assert_less(r2, 0.33, "sum should be less than %s" % 0.33)


if __name__ == '__main__':
    unittest.main()
