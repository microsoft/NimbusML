# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import unittest

import numpy as np
from nimbusml.datasets import get_dataset
from nimbusml.decomposition import PcaAnomalyDetector
from sklearn.model_selection import train_test_split
from sklearn.utils.testing import assert_almost_equal


class TestPcaAnomalyDetector(unittest.TestCase):

    def test_PcaAnomalyDetector(self):
        np.random.seed(0)
        df = get_dataset("iris").as_df()
        df.drop(['Species'], inplace=True, axis=1)
        X_train, X_test, y_train, y_test = train_test_split(
            df.loc[:, df.columns != 'Label'], df['Label'])
        svm = PcaAnomalyDetector(rank=3)
        svm.fit(X_train, y_train)
        scores = svm.predict(X_test)
        assert_almost_equal(
            scores.sum().sum(),
            4.181632,
            decimal=7,
            err_msg="Sum should be %s" %
                    4.181632)


if __name__ == '__main__':
    unittest.main()
