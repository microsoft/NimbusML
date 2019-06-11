# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
from inspect import getargspec

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.testing import assert_greater


def check_supported_losses(testcase, learner, losses, acc_threshold):
    # REVIEW: We are leaving out many cases by not doing this in
    # get_accuracy(), but this covers
    # the tests that are currently failing. Modifying get_accuracy() will
    # require changing the param
    # type of learner and thus all calls to get_accuracy(), so I created bug
    # 247514 for that work.
    learner_args = getargspec(learner.__init__).args
    kwargs = {}
    if 'number_of_threads' in learner_args and 'shuffle' in learner_args:
        kwargs.update({'number_of_threads': 1, 'shuffle': False})
    for l in losses:
        kwargs['loss'] = l
        accuracy = get_accuracy(testcase, learner(**kwargs))
        assert_greater(
            accuracy, acc_threshold,
            "Accuracy of %s was lower than expected %s." %
            (accuracy, acc_threshold))


def check_unsupported_losses(testcase, learner, losses):
    for l in losses:
        msg = '{} is not supported, but exception was not thrown.'.format(l)
        with testcase.assertRaises(TypeError, msg=msg):
            learner(loss=l)


def get_accuracy(testcase, learner):
    np.random.seed(0)
    X_train, X_test, y_train, y_test = train_test_split(testcase.X, testcase.y)
    lr = learner.fit(X_train, y_train)
    scores = lr.predict(X_test)
    if str(y_test.dtype) == 'category':
        assert scores.dtype == 'O'
        accuracy = np.mean(scores == [str(i) for i in y_test])
    else:
        accuracy = np.mean(y_test == [i for i in scores])
    return accuracy


def split_features_and_label(df, label_name):
    features = df.loc[:, df.columns != label_name]
    label = df[label_name]
    return features, label
