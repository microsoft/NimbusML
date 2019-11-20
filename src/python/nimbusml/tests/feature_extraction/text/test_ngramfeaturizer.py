# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import os
import unittest

import numpy as np
import sys
np.set_printoptions(threshold=np.inf)
import six
from nimbusml.datasets import get_dataset
from nimbusml.feature_extraction.text import NGramFeaturizer
from nimbusml.internal.entrypoints._ngramextractor_ngram import n_gram
from nimbusml.utils import get_X_y
from sklearn.model_selection import train_test_split
from sklearn.utils.testing import assert_equal


class TestNGramFeaturizer(unittest.TestCase):

    def test_ngramfeaturizer(self):
        np.random.seed(0)
        train_file = get_dataset('wiki_detox_train').as_filepath()
        (train,
         label) = get_X_y(train_file,
                          label_column='Sentiment',
                          sep='\t',
                          encoding="utf-8")
        X_train, X_test, y_train, y_test = train_test_split(
            train['SentimentText'], label)

        # map text reviews to vector space
        texttransform = NGramFeaturizer(
            word_feature_extractor=n_gram(),
            vector_normalizer='None') << 'SentimentText'
        print("X_train Before Just Fit Column Names Start\n")
        X_train_fit = texttransform.fit(X_train[:100])
        print(X_train_fit)
        print("X_train Before Just Fit Column Names End\n")
        print("X_train Before Just Transform Column Names Start\n")
        X_train_transform = X_train_fit.transform(X_train[:100])
        for col in X_train_transform.columns:
            print(col)
        print("X_train Before Just Transform Column Names End\n")
        print("Len of X_train_transform: {}".format(len(X_train_transform)))
        #X_train = texttransform.fit_transform(X_train[:100])


        #print("X_train Column Names Start\n")
        #for col in X_train.columns:
        #    print(col)
        #print("X_train Column Names End\n")
        print("X_train_transform.iloc[:].sum()\n")
        print(X_train_transform.iloc[:].sum())
        sum = X_train_transform.iloc[:].sum().sum()
        print("Sum")
        print(sum)
        assert_equal(sum, 30513, "sum of all features is incorrect!")


if __name__ == '__main__':
    unittest.main()
