# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import unittest

from nimbusml.datasets import get_dataset
from nimbusml.linear_model import FastLinearClassifier
from nimbusml.tests.test_utils import split_features_and_label


def get_iris():
    df = get_dataset("iris").as_df()
    df.drop(['Label'], inplace=True, axis=1)
    df['Label'] = df['Species']
    df.drop(['Species'], inplace=True, axis=1)
    df.drop(['Setosa'], inplace=True, axis=1)
    X, y = split_features_and_label(df, 'Label')
    return X, y


class TestTextLabel(unittest.TestCase):

    def test_text_label(self):
        X, y = get_iris()
        ap = FastLinearClassifier(
            feature=[
                'Sepal_Width',
                'Sepal_Length',
                'Petal_Width',
                'Petal_Length'])
        ap.fit(X, y)
        scores = ap.predict(X)
        assert str(scores.dtype) == "object"


if __name__ == '__main__':
    unittest.main()
