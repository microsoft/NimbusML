# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import unittest

try:
    # pandas 0.20.0+
    from pandas.api.types import is_string_dtype
except ImportError:
    def is_string_dtype(dt):
        return 'object' in str(dt) or "dtype('O')" in str(dt)

import numpy as np
from nimbusml.feature_extraction.categorical import OneHotVectorizer
from nimbusml.linear_model import LinearSvmBinaryClassifier
from nimbusml.datasets import get_dataset
from nimbusml import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.utils.testing import assert_greater


class TestLinearSvmBinaryClassifier(unittest.TestCase):

    def test_linearsvm(self):
        np.random.seed(0)
        df = get_dataset("infert").as_df()
        # remove : and ' ' from column names, and encode categorical column
        df.columns = [i.replace(': ', '') for i in df.columns]
        assert is_string_dtype(df['education_str'].dtype)
        df = (OneHotVectorizer() << ['education_str']).fit_transform(df)
        assert 'education_str' not in df.columns
        X_train, X_test, y_train, y_test = train_test_split(
            df.loc[:, df.columns != 'case'], df['case'], random_state=0)
        svm = LinearSvmBinaryClassifier(shuffle=False).fit(X_train, y_train)
        scores = svm.predict(X_test)
        accuracy = np.mean(y_test == [i for i in scores])
        assert_greater(accuracy, 0.96, "accuracy should be %s" % 0.96)


if __name__ == '__main__':
    unittest.main()

