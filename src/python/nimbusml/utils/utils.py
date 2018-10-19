# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
"""
general utility functions
"""

import os

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
from sklearn.utils.testing import assert_greater


# select columns from DataFrame insize a pipeline


class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns, ravel=False):
        self.columns = columns
        self.ravel = ravel

    def fit(self, x, y=None):
        return self

    def transform(self, df, drop=False):
        if drop:
            df = df.drop[self.columns]
        else:
            df = df[self.columns]
        if self.ravel:
            return np.ravel(df[self.columns].values)
        else:
            return df[self.columns]


def load_img(filename):
    from imageio import imread
    from scipy.misc import imresize
    with open(filename, "r") as f:
        N = sum(1 for line in f)

    H = 32
    W = 32
    C = 3
    X = np.zeros((N, H, W, C))
    y = np.zeros(N, dtype=np.uint8)
    base_path = os.path.dirname(filename)
    with open(filename, "r") as f:
        line = f.readline().strip()
        i = 0
        while line:
            #            if (i % 10 == 0):
            #                print(i)

            relative_path, label = line.split('\t')
            full_path = os.path.join(base_path, relative_path)
            #            print("{0} {1}".format(full_path, label))

            img = imread(full_path)
            img_resize = imresize(img, (H, W))

            X[i] = img_resize.astype(np.float64)
            y[i] = int(label)

            line = f.readline().strip()
            i += 1

    return X, y


# extract label and features from train/test fiels
def get_X_y(filename, label_column=None, label_index=None, header=0, sep='\t',
            nrows=None, features=None, names=None,
            encoding="iso-8859-1"):  # default encoding = "ISO-8859-1"
    df = pd.read_csv(
        filename,
        header=header,
        sep=sep,
        nrows=nrows,
        error_bad_lines=False,
        encoding=encoding,
        names=names)
    if header is not None:
        df.rename(columns=lambda x: x.strip(), inplace=True)
    else:
        df.columns = ['V' + str(x) for x in range(len(df.columns))]
    X = None
    y = None
    if label_column is not None:
        X = df.loc[:, df.columns != label_column]
        if features is not None:
            X = X[features]
        y = df.loc[:, df.columns == label_column]
    elif label_index is not None:
        X = df.ix[:, [i for i in range(len(df.columns)) if i != label_index]]
        y = df.ix[:, [label_index]]
    return X, y


def evaluate_binary_classifier(target, predicted, probabilities=None):
    accuracy = np.mean(target == predicted)
    auc_score = None
    if probabilities is not None:
        auc_score = roc_auc_score(target, probabilities)
    return (accuracy, auc_score)


def check_accuracy(test_file, label_column, predictions, threshold, sep=','):
    (test, label) = get_X_y(test_file, label_column, sep=sep)
    accuracy = np.mean(label[label_column].values ==
                       predictions.ix[:, 'PredictedLabel'].values)
    assert_greater(
        accuracy,
        threshold,
        "accuracy should be greater than %s" %
        threshold)


def check_accuracy_scikit(
        test_file,
        label_column,
        predictions,
        threshold,
        sep=','):
    (test, label) = get_X_y(test_file, label_column, sep=sep)
    accuracy = np.mean(label[label_column].values == predictions.values)
    assert_greater(
        accuracy,
        threshold,
        "accuracy should be greater than %s" %
        threshold)
