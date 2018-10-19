# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import unittest

from nimbusml.datasets import get_dataset
from nimbusml.feature_extraction.categorical import OneHotVectorizer
from nimbusml.linear_model import SgdBinaryClassifier
from nimbusml.loss import Exp, Hinge, Poisson, SmoothedHinge, Squared, Tweedie
from nimbusml.tests.test_utils import get_accuracy, \
    split_features_and_label, \
    check_supported_losses, check_unsupported_losses
from sklearn.utils.testing import assert_greater


class TestSgdBinaryClassifier(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        df = get_dataset("infert").as_df()
        # remove : and ' ' from column names, and encode categorical column
        df.columns = [i.replace(': ', '') for i in df.columns]
        df = (OneHotVectorizer() << ['education_str']).fit_transform(df)
        cls.X, cls.y = split_features_and_label(df, 'case')

    def test_sgdbinaryclassifier(self):
        accuracy = get_accuracy(self, SgdBinaryClassifier())
        assert_greater(accuracy, 0.87, "accuracy should be %s" % 0.87)

    def test_sgdbinaryclassifier_supported_losses(self):
        losses = [
            'exp', 'log', 'hinge', 'smoothed_hinge', Exp(
                beta=0.9), Hinge(
                margin=0.5), SmoothedHinge(
                smoothing_const=0.5)]
        check_supported_losses(self, SgdBinaryClassifier, losses, 0.87)

    def test_sgdbinaryclassifier_unsupported_losses(self):
        losses = [
            'squared',
            'poisson',
            'tweedie',
            Squared(),
            Poisson(),
            Tweedie(),
            100,
            'random_str']
        check_unsupported_losses(self, SgdBinaryClassifier, losses)


if __name__ == '__main__':
    unittest.main()
