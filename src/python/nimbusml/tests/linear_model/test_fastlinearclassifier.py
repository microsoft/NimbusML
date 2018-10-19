# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import unittest

from nimbusml.datasets import get_dataset
from nimbusml.linear_model import FastLinearClassifier
from nimbusml.loss import Exp, Hinge, Poisson, SmoothedHinge, Squared, Tweedie
from nimbusml.tests.test_utils import get_accuracy, \
    split_features_and_label, \
    check_supported_losses, check_unsupported_losses
from sklearn.utils.testing import assert_almost_equal


class TestFastLinearClassifier(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        df = get_dataset("iris").as_df()
        df.drop(['Species'], inplace=True, axis=1)
        cls.X, cls.y = split_features_and_label(df, 'Label')

    def test_FastLinearClassifier(self):
        acc = get_accuracy(self, FastLinearClassifier())
        assert_almost_equal(
            acc,
            0.97368421052,
            decimal=8,
            err_msg="Sum should be %s" %
                    0.97368421052)

    def test_FastLinearClassifier_supported_losses(self):
        losses = [
            'log', 'hinge', 'smoothed_hinge', Hinge(
                margin=0.5), SmoothedHinge(
                smoothing_const=0.5)]
        # Expected accuracy when single threaded, no shuffle: 0.97368421052
        check_supported_losses(self, FastLinearClassifier, losses, 0.94)

    def test_FastLinearClassifier_unsupported_losses(self):
        losses = ['exp', 'squared', 'poisson', 'tweedie', Exp(
            beta=0.5), Squared(), Poisson(), Tweedie(), 100, 'random_str']
        check_unsupported_losses(self, FastLinearClassifier, losses)


if __name__ == '__main__':
    unittest.main()
