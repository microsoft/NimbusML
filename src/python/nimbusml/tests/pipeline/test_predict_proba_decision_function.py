# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
import os
import tempfile
import unittest

import numpy as np
from nimbusml import Pipeline
from nimbusml.datasets import get_dataset
from nimbusml.decomposition import FactorizationMachineBinaryClassifier
from nimbusml.ensemble import LightGbmClassifier
from nimbusml.linear_model import FastLinearClassifier
from nimbusml.linear_model import LogisticRegressionBinaryClassifier
from nimbusml.linear_model import LogisticRegressionClassifier
from nimbusml.naive_bayes import NaiveBayesClassifier
from nimbusml.tests.test_utils import split_features_and_label
from sklearn.model_selection import train_test_split
from sklearn.utils.testing import assert_almost_equal, assert_equal

# use iris dataset
np.random.seed(0)
df = get_dataset("iris").as_df()
df.drop(['Species'], inplace=True, axis=1)
df.Label = [1 if x == 1 else 0 for x in df.Label]
features, labels = split_features_and_label(df, 'Label')
X_train, X_test, y_train, y_test = \
    train_test_split(features, labels)

# 3 class dataset with integer labels
np.random.seed(0)
df = get_dataset("iris").as_df()
df.drop(['Species'], inplace=True, axis=1)
features_3class_int, labels_3class_int = split_features_and_label(df, 'Label')
X_train_3class_int, X_test_3class_int, y_train_3class_int, y_test_3class_int = \
    train_test_split(features_3class_int, labels_3class_int)

# 3 class dataset with string labels
np.random.seed(0)
df = get_dataset("iris").as_df()
df.drop(['Species'], inplace=True, axis=1)
_str_map = {0: 'Red', 1: 'Green', 2: 'Blue'}
df.Label = df.Label.apply(lambda x: _str_map[x])
features_3class, labels_3class = split_features_and_label(df, 'Label')
X_train_3class, X_test_3class, y_train_3class, y_test_3class = \
    train_test_split(features_3class, labels_3class)


# fit classifier, return sum of probabilites
def proba_sum(pipe_or_clf):
    pipe_or_clf.fit(X_train, y_train)
    return pipe_or_clf.predict_proba(X_test).sum()


# fit classifier, return sum of decision values
def decfun_sum(pipe_or_clf):
    pipe_or_clf.fit(X_train, y_train)
    return pipe_or_clf.decision_function(X_test).sum()


def check_unsupported_predict_proba(testcase, clf, X_train, y_train, X_test):
    msg = '{} doesnt support predict_proba(), but exception was not ' \
          'thrown.'.format(clf)
    with testcase.assertRaises(AttributeError, msg=msg):
        clf.fit(X_train, y_train)
        clf.predict_proba(X_test)


def check_unsupported_decision_function(
        testcase, clf, X_train, y_train, X_test):
    msg = '{} doesnt support decision_function(), but exception was not ' \
          'thrown.'.format(clf)
    with testcase.assertRaises(AttributeError, msg=msg):
        clf.fit(X_train, y_train)
        clf.decision_function(X_test)


invalid_predict_proba_output = "Invalid sum of probabilities from predict_prob"


class TestPredictProba(unittest.TestCase):
    def test_pass_predict_proba_binary(self):
        assert_almost_equal(
            proba_sum(
                LogisticRegressionBinaryClassifier(
                    number_of_threads=1)),
            38.0,
            decimal=3,
            err_msg=invalid_predict_proba_output)

    def test_pass_predict_proba_binary_with_pipeline(self):
        assert_almost_equal(
            proba_sum(Pipeline([LogisticRegressionBinaryClassifier(
                number_of_threads=1)])), 38.0, decimal=3,
            err_msg=invalid_predict_proba_output)

    def test_pass_predict_proba_multiclass(self):
        assert_almost_equal(
            proba_sum(
                LogisticRegressionClassifier()),
            38.0,
            decimal=3,
            err_msg=invalid_predict_proba_output)

    def test_pass_predict_proba_multiclass_with_pipeline(self):
        algos = [
            LogisticRegressionClassifier(),
            FastLinearClassifier(),
            LightGbmClassifier()]
        for algo in algos:
            assert_almost_equal(proba_sum(Pipeline([algo])), 38.0, decimal=3,
                                err_msg=invalid_predict_proba_output)

    def test_pass_predict_proba_multiclass_3class(self):
        clf = FastLinearClassifier(number_of_threads=1)
        clf.fit(X_train_3class, y_train_3class)
        s = clf.predict_proba(X_test_3class).sum()
        assert_almost_equal(
            s,
            38.0,
            decimal=4,
            err_msg=invalid_decision_function_output)
        assert_equal(set(clf.classes_), {'Blue', 'Green', 'Red'})

    def test_pass_predict_proba_multiclass_with_pipeline_adds_classes(self):
        clf = FastLinearClassifier(number_of_threads=1)
        pipeline = Pipeline([clf])
        pipeline.fit(X_train_3class, y_train_3class)

        expected_classes = {'Blue', 'Green', 'Red'}
        assert_equal(set(clf.classes_), expected_classes)
        assert_equal(set(pipeline.classes_), expected_classes)

        s = pipeline.predict_proba(X_test_3class).sum()
        assert_almost_equal(
            s,
            38.0,
            decimal=4,
            err_msg=invalid_decision_function_output)

        assert_equal(set(clf.classes_),  expected_classes)
        assert_equal(set(pipeline.classes_),  expected_classes)

    def test_pass_predict_proba_multiclass_3class_retains_classes_type(self):
        clf = FastLinearClassifier(number_of_threads=1)
        clf.fit(X_train_3class_int, y_train_3class_int)
        s = clf.predict_proba(X_test_3class_int).sum()
        assert_almost_equal(
            s,
            38.0,
            decimal=4,
            err_msg=invalid_decision_function_output)
        assert_equal(set(clf.classes_), {0, 1, 2})

    def test_fail_predict_proba_multiclass_with_pipeline(self):
        check_unsupported_predict_proba(self, Pipeline(
            [NaiveBayesClassifier()]), X_train, y_train, X_test)

    def test_pass_predict_proba_from_load_model(selfs):
        pipeline = Pipeline([LogisticRegressionBinaryClassifier()])
        pipeline.fit(X_train, y_train)
        probs1 = pipeline.predict_proba(X_test)
        sum1 = probs1.sum().sum()
        (fd, modelfilename) = tempfile.mkstemp(suffix='.model.bin')
        fl = os.fdopen(fd, 'w')
        fl.close()
        pipeline.save_model(modelfilename)

        pipeline2 = Pipeline()
        pipeline2.load_model(modelfilename)
        probs2 = pipeline2.predict_proba(X_test)
        sum2 = probs2.sum().sum()
        assert_equal(
            sum1,
            sum2,
            "model probabilities don't match after loading model")


invalid_decision_function_output = 'Invalid sum of scores ' \
                                   'from decision_function'


class TestDecisionFunction(unittest.TestCase):
    def test_pass_decision_function_binary(self):
        assert_almost_equal(decfun_sum(FactorizationMachineBinaryClassifier(
        )), -32.618393, decimal=5, err_msg=invalid_decision_function_output)

    def test_pass_decision_function_binary_with_pipeline(self):
        assert_almost_equal(
            decfun_sum(Pipeline([FactorizationMachineBinaryClassifier(
            )])), -32.618393, decimal=5,
            err_msg=invalid_decision_function_output)

    def test_pass_decision_function_multiclass(self):
        assert_almost_equal(decfun_sum(NaiveBayesClassifier()), -
                            96.87325, decimal=4,
                            err_msg=invalid_decision_function_output)

    def test_pass_decision_function_multiclass_with_pipeline(self):
        assert_almost_equal(decfun_sum(Pipeline([NaiveBayesClassifier(
        )])), -96.87325, decimal=4, err_msg=invalid_decision_function_output)

    def test_pass_decision_function_multiclass_3class(self):
        clf = FastLinearClassifier(number_of_threads=1)
        clf.fit(X_train_3class, y_train_3class)
        s = clf.decision_function(X_test_3class).sum()
        assert_almost_equal(
            s,
            38.0,
            decimal=4,
            err_msg=invalid_decision_function_output)
        assert_equal(set(clf.classes_), {'Blue', 'Green', 'Red'})

    def test_pass_decision_function_multiclass_with_pipeline_adds_classes(self):
        clf = FastLinearClassifier(number_of_threads=1)
        pipeline = Pipeline([clf])
        pipeline.fit(X_train_3class, y_train_3class)

        expected_classes = {'Blue', 'Green', 'Red'}
        assert_equal(set(clf.classes_), expected_classes)
        assert_equal(set(pipeline.classes_), expected_classes)

        s = pipeline.decision_function(X_test_3class).sum()
        assert_almost_equal(
            s,
            38.0,
            decimal=4,
            err_msg=invalid_decision_function_output)

        assert_equal(set(clf.classes_), expected_classes)
        assert_equal(set(pipeline.classes_), expected_classes)

    def test_pass_decision_function_multiclass_3class_retains_classes_type(self):
        clf = FastLinearClassifier(number_of_threads=1)
        clf.fit(X_train_3class_int, y_train_3class_int)
        s = clf.decision_function(X_test_3class_int).sum()
        assert_almost_equal(
            s,
            38.0,
            decimal=4,
            err_msg=invalid_decision_function_output)
        assert_equal(set(clf.classes_), {0, 1, 2})

    def test_fail_decision_function_multiclass(self):
        check_unsupported_decision_function(
            self, LogisticRegressionClassifier(), X_train, y_train, X_test)

    def test_fail_decision_function_multiclass_with_pipeline(self):
        check_unsupported_decision_function(self, Pipeline(
            [LogisticRegressionClassifier()]), X_train, y_train, X_test)


if __name__ == '__main__':
    unittest.main()
