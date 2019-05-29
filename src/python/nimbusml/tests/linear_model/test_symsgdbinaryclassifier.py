# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import unittest
import os

import numpy as np
from nimbusml.datasets import get_dataset
from nimbusml.feature_extraction.categorical import OneHotVectorizer
from nimbusml.linear_model import SymSgdBinaryClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils.testing import assert_greater


class TestSymSgdBinaryClassifier(unittest.TestCase):

    @unittest.skipIf(os.name != "nt", "BUG: SymSgd lib fails to load on Linux")
    def test_SymSgdBinaryClassifier(self):
        np.random.seed(0)
        df = get_dataset("infert").as_df()
        df.columns = [i.replace(': ', '') for i in df.columns]
        df = (OneHotVectorizer() << 'education_str').fit_transform(df)
        X_train, X_test, y_train, y_test = train_test_split(
            df.loc[:, df.columns != 'case'], df['case'], random_state=0)
        lr = SymSgdBinaryClassifier(
            shuffle=False, number_of_threads=1).fit(
            X_train, y_train)
        scores = lr.predict(X_test)
        acc = np.mean(y_test == [i for i in scores])
        # Removing randomness (shuffle=False) may be worse
        # because classes are not well distributed.
        assert_greater(acc, 0.25, "accuracy should be around %s" % 0.65)


if __name__ == '__main__':
    unittest.main()
