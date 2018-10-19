# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import unittest

import numpy as np
from nimbusml.datasets import get_dataset
from nimbusml.ensemble import FastTreesTweedieRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.utils.testing import assert_greater, assert_less


class TestFastTreesTweedieRegressor(unittest.TestCase):

    def test_fasttreestweedieregressor(self):
        np.random.seed(0)

        df = get_dataset("airquality").as_df().fillna(0)
        df = df[df.Ozone.notnull()]

        X_train, X_test, y_train, y_test = train_test_split(
            df.loc[:, df.columns != 'Ozone'], df['Ozone'])

        # Train a model and score
        ftree = FastTreesTweedieRegressor().fit(X_train, y_train)
        scores = ftree.predict(X_test)

        r2 = r2_score(y_test, scores)
        assert_greater(r2, 0.479, "sum should be greater than %s" % 0.479)
        assert_less(r2, 0.480, "sum should be less than %s" % 0.480)


if __name__ == '__main__':
    unittest.main()
