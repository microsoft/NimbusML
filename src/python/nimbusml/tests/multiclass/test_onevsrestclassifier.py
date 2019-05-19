# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
import platform
import unittest

import numpy as np
from nimbusml import Pipeline
from nimbusml.datasets import get_dataset
from nimbusml.ensemble import GamBinaryClassifier, \
    FastTreesBinaryClassifier, \
    FastForestBinaryClassifier
from nimbusml.ensemble import LightGbmBinaryClassifier
from nimbusml.linear_model import FastLinearBinaryClassifier, \
    AveragedPerceptronBinaryClassifier
from nimbusml.linear_model import LogisticRegressionBinaryClassifier, \
    SgdBinaryClassifier
# from nimbusml.linear_model import SymSgdBinaryClassifier
from nimbusml.multiclass import OneVsRestClassifier
from nimbusml.tests.test_utils import split_features_and_label
from sklearn.model_selection import train_test_split
from sklearn.utils.testing import assert_equal, assert_not_equal, \
    assert_greater

# use iris dataset
np.random.seed(0)
df = get_dataset("iris").as_df()
df.drop(['Species'], inplace=True, axis=1)
features, labels = split_features_and_label(df, 'Label')
X_train, X_test, y_train, y_test = \
    train_test_split(features, labels)


# fit classifier, return sum of probabilites
def proba_average(ovr):
    ovr.fit(X_train, y_train)
    return ovr.predict_proba(X_test).sum(axis=1).mean()


# fit classifier, return sum of decision values
def decfun_average(ovr):
    ovr.fit(X_train, y_train)
    return ovr.decision_function(X_test).sum(axis=1).mean()


def accuracy(ovr):
    pipe = Pipeline([ovr])
    pipe.fit(X_train, y_train)
    metrics, _ = pipe.test(X_train, y_train)
    return metrics


def check_predict_proba_when_trained_with_use_probabilites_false(
        testcase,
        ovr,
        clf):
    msg = '{} + OVA  + use_probabilities=False should throw exception upon ' \
          'calling predict_proba(), ' \
          'but exception was not thrown.'.format(clf)
    with testcase.assertRaises(ValueError, msg=msg):
        ovr.fit(X_train, y_train)
        ovr.predict_proba(X_test)


def check_decision_function_when_trained_with_use_probabilites_true(
        testcase,
        ovr,
        clf):
    msg = '{} + OVA  + use_probabilities=True should throw exception upon ' \
          'calling decision_function(), ' \
          'but exception was not thrown.'.format(clf)
    with testcase.assertRaises(ValueError, msg=msg):
        ovr.fit(X_train, y_train)
        ovr.decision_function(X_test)


class TestOneVsRestClassfier(unittest.TestCase):
    def test_predict_proba_produces_distribution_sum_to_1(self):
        clfs = [
            # TODO: BUG 231482 , why doesnt FM work
            # FactorizationMachineBinaryClassifier(),
            LogisticRegressionBinaryClassifier(),
            FastForestBinaryClassifier(minimum_example_count_per_leaf=1),
            GamBinaryClassifier(),
            AveragedPerceptronBinaryClassifier(),
            FastTreesBinaryClassifier(minimum_example_count_per_leaf=1),
            LightGbmBinaryClassifier(),
            FastLinearBinaryClassifier(),
            SgdBinaryClassifier(),
            # TODO: why symsgd does not sum to 1.0
            # SymSgdBinaryClassifier(),
        ]

        for clf in clfs:
            ovr = OneVsRestClassifier(classifier=clf)
            probmean = proba_average(ovr)
            assert_equal(
                probmean,
                1.0,
                '{} probabilites {} do not sum to 1.0 over 3 classes'.format(
                    clf.__class__,
                    probmean))

    def test_failing_predict_proba_called_with_use_probabilites_false(self):
        clfs = [
            # TODO: BUG 231482 , why doesnt FM work
            # FactorizationMachineBinaryClassifier(),
            LogisticRegressionBinaryClassifier(),
            FastForestBinaryClassifier(minimum_example_count_per_leaf=1),
            GamBinaryClassifier(),
            AveragedPerceptronBinaryClassifier(),
            FastTreesBinaryClassifier(minimum_example_count_per_leaf=1),
            LightGbmBinaryClassifier(),
            FastLinearBinaryClassifier(),
            SgdBinaryClassifier(),
            # SymSgdBinaryClassifier(),
        ]

        for clf in clfs:
            ovr = OneVsRestClassifier(classifier=clf, use_probabilities=False)
            check_predict_proba_when_trained_with_use_probabilites_false(
                self, ovr, clf)

    def test_decision_function_produces_distribution_not_sum_to_1(self):
        clfs = [
            # TODO: BUG 231482 , why doesnt FM work
            # FactorizationMachineBinaryClassifier(),
            LogisticRegressionBinaryClassifier(),
            FastForestBinaryClassifier(minimum_example_count_per_leaf=1),
            GamBinaryClassifier(),
            AveragedPerceptronBinaryClassifier(),
            FastTreesBinaryClassifier(minimum_example_count_per_leaf=1),
            LightGbmBinaryClassifier(),
            FastLinearBinaryClassifier(),
            SgdBinaryClassifier(),
            # SymSgdBinaryClassifier(),
        ]

        for clf in clfs:
            ovr = OneVsRestClassifier(classifier=clf, use_probabilities=False)
            scoremean = decfun_average(ovr)
            assert_not_equal(
                scoremean,
                1.0,
                '{} raw scores should not sum to 1.0 over 3 classes'.format(
                    clf.__class__))

    def test_failing_decision_function_called_with_use_probabilites_true(self):
        clfs = [
            # TODO: BUG 231482 , why doesnt FM work
            # FactorizationMachineBinaryClassifier(),
            LogisticRegressionBinaryClassifier(),
            FastForestBinaryClassifier(minimum_example_count_per_leaf=1),
            GamBinaryClassifier(),
            AveragedPerceptronBinaryClassifier(),
            FastTreesBinaryClassifier(minimum_example_count_per_leaf=1),
            LightGbmBinaryClassifier(),
            FastLinearBinaryClassifier(),
            SgdBinaryClassifier(),
            # SymSgdBinaryClassifier(),
        ]

        for clf in clfs:
            ovr = OneVsRestClassifier(classifier=clf, use_probabilities=True)
            check_decision_function_when_trained_with_use_probabilites_true(
                self, ovr, clf)

    def test_ovr_accuracy(self):
        clfs = [
            # TODO: BUG 231482 , why doesnt FM work
            # FactorizationMachineBinaryClassifier(),
            LogisticRegressionBinaryClassifier(number_of_threads=1),
            FastForestBinaryClassifier(minimum_example_count_per_leaf=1, number_of_threads=1),
            GamBinaryClassifier(number_of_threads=1),
            AveragedPerceptronBinaryClassifier(),
            FastTreesBinaryClassifier(minimum_example_count_per_leaf=1, number_of_threads=1),
            FastLinearBinaryClassifier(number_of_threads=1),
            SgdBinaryClassifier(number_of_threads=1),
            # SymSgdBinaryClassifier(number_of_threads=1),
        ]

        for clf in clfs:
            ovr = OneVsRestClassifier(classifier=clf, use_probabilities=True)
            metrics = accuracy(ovr)
            accu = metrics['Accuracy(micro-avg)'][0]
            # algos will have wide range of accuracy, so use low bar. Also
            # checks Pipeline + Ova + clf
            assert_greater(
                accu, 0.65, "{} accuracy is too low {}".format(
                    clf.__class__, accu))


if __name__ == '__main__':
    unittest.main()
