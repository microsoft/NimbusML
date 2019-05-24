# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
"""
test sparse
"""
import unittest

import numpy as np
import pandas as pd
from nimbusml.linear_model import FastLinearBinaryClassifier
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.testing import assert_raises

train_reviews = pd.DataFrame(
    data=dict(
        review=[
            "This is great",
            "I hate it",
            "Love it",
            "Do not like it",
            "Really like it",
            "I hate it",
            "I like it a lot",
            "I kind of hate it",
            "I do like it",
            "I really hate it",
            "It is very good",
            "I hate it a bunch",
            "I love it a bunch",
            "I hate it",
            "I like it very much",
            "I hate it very much.",
            "I really do love it",
            "I really do hate it",
            "Love it!",
            "Hate it!",
            "I love it",
            "I hate it",
            "I love it",
            "I hate it",
            "I love it"],
        like=[
            True,
            False,
            True,
            False,
            True,
            False,
            True,
            False,
            True,
            False,
            True,
            False,
            True,
            False,
            True,
            False,
            True,
            False,
            True,
            False,
            True,
            False,
            True,
            False,
            True]))

test_reviews = pd.DataFrame(
    data=dict(
        review=[
            "This is great",
            "I hate it",
            "Love it",
            "Really like it",
            "I hate it",
            "I like it a lot",
            "I love it",
            "I do like it",
            "I really hate it",
            "I love it"]))

train_X = train_reviews['review']
train_y = np.ravel(train_reviews['like'])
test_X = test_reviews['review']


def test_dtype(xtype=None, ytype=None, dense=False):
    tfidf = TfidfVectorizer(
        analyzer='word',
        ngram_range=(
            1,
            1),
        min_df=2,
        max_df=0.8,
        norm='l2')
    sparse_data = tfidf.fit_transform(train_X, train_y)
    assert isinstance(sparse_data, csr_matrix)
    if xtype is not None:
        sparse_data = csr_matrix(sparse_data, dtype=xtype)
        assert sparse_data.dtype == xtype

    xdata = sparse_data
    if dense is True:
        xdata = sparse_data.todense()
        assert xdata.dtype == xtype

    ydata = train_y
    if ytype is not None:
        ydata = ydata.astype(ytype)
        assert ydata.dtype == ytype

    algo = FastLinearBinaryClassifier(maximum_number_of_iterations=2)
    algo.fit(xdata, ydata)
    assert algo.model_ is not None

    test_sparse_data = tfidf.transform(test_X)
    assert isinstance(test_sparse_data, csr_matrix)
    if xtype is not None:
        test_sparse_data = csr_matrix(test_sparse_data, dtype=xtype)
        assert test_sparse_data.dtype == xtype

    xtest_data = test_sparse_data
    if dense is True:
        xtest_data = test_sparse_data.todense()
        assert xtest_data.dtype == xtype

    data = algo.predict(xtest_data)
    assert data.size == test_reviews.size


class TestDTypes(unittest.TestCase):
    def test_data_types(self):
        types = [
            None,
            np.bool,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
            np.double,
            np.float,
            np.float16
        ]
        for xtype in types:
            for ytype in types:
                print(
                    "================ Testing sparse xtype %s, ytype %s "
                    "================" %
                    (str(xtype), str(ytype)))
                if (xtype == np.uint64 or xtype == np.float16 or ytype == np.float16):
                    assert_raises(
                        (TypeError, ValueError, RuntimeError), test_dtype,
                        xtype, ytype)
                else:
                    test_dtype(xtype, ytype)
                print(
                    "================ Testing dense xtype %s, ytype %s "
                    "================" %
                    (str(xtype), str(ytype)))
                if (xtype == np.float16):
                    assert_raises(
                        ValueError, test_dtype, xtype, ytype, dense=True)
                elif ytype == np.float16:
                    assert_raises((TypeError, ValueError, RuntimeError),
                                  test_dtype, xtype, ytype, dense=True)
                else:
                    test_dtype(xtype, ytype, dense=True)


if __name__ == "__main__":
    unittest.main()
