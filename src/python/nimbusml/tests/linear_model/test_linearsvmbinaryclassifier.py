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
from sklearn.utils.testing import assert_almost_equal, assert_greater


class TestLinearSvmBinaryClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        np.random.seed(0)
        df = get_dataset("infert").as_df()
        # remove : and ' ' from column names, and encode categorical column
        df.columns = [i.replace(': ', '') for i in df.columns]
        assert is_string_dtype(df['education_str'].dtype)
        df = (OneHotVectorizer() << ['education_str']).fit_transform(df)
        assert 'education_str' not in df.columns
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(df.loc[:, df.columns != 'case'],
                             df['case'],
                             random_state=0)
        self.svm = LinearSvmBinaryClassifier(shuffle=False).fit(self.X_train,
                                                                self.y_train)
        self.predictions = self.svm.predict(self.X_test)
        self.accuracy = np.mean(self.y_test == [i for i in self.predictions])

    def test_linearsvm(self):
        assert_greater(self.accuracy, 0.96, "accuracy should be %s" % 0.96)

    def test_linearsvm_predict_proba(self):
        probabilities = self.svm.predict_proba(self.X_test)
        # Test that the class probabilities for each instance add up to 1
        [assert_almost_equal(probabilities[i][0] + probabilities[i][1], 1) \
            for i in range(probabilities.shape[0])]

    def test_linearsvm_decision_function(self):
        fn = self.svm.decision_function(self.X_test)
        predictions_from_fn = [fn[i] >= 0 for i in range(len(fn))]
        assert [predictions_from_fn[i] == self.predictions[i] \
            for i in range(len(self.predictions))]
        accuracy_from_fn = np.mean(
            self.y_test == [i for i in predictions_from_fn])
        assert accuracy_from_fn == self.accuracy


if __name__ == '__main__':
    unittest.main()

