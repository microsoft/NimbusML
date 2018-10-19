# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
import platform
import unittest

import numpy as np
from nimbusml.datasets import get_dataset
from nimbusml.ensemble import LightGbmBinaryClassifier
from nimbusml.feature_extraction.categorical import OneHotVectorizer
from sklearn.model_selection import train_test_split
from sklearn.utils.testing import assert_greater


class TestLightGbmBinaryClassifier(unittest.TestCase):

    def test_lightgbmbinaryclassifier(self):
        np.random.seed(0)

        df = get_dataset("infert").as_df()

        # remove : and ' ' from column names, and encode categorical column
        df.columns = [i.replace(': ', '') for i in df.columns]
        df = (OneHotVectorizer() << 'education_str').fit_transform(df)

        X_train, X_test, y_train, y_test = \
            train_test_split(df.loc[:, df.columns != 'case'], df['case'])

        lr = LightGbmBinaryClassifier().fit(X_train, y_train)
        scores = lr.predict(X_test)
        accuracy = np.mean(y_test == [i for i in scores])
        assert_greater(accuracy, 0.98, "accuracy should be %s" % 0.98)


if __name__ == '__main__':
    unittest.main()
