# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import os
import unittest

import numpy as np
import six
from nimbusml.datasets import get_dataset
from nimbusml.feature_extraction.text import NGramFeaturizer
from nimbusml.internal.entrypoints._ngramextractor_ngram import n_gram
from nimbusml.utils import get_X_y
from sklearn.model_selection import train_test_split
from sklearn.utils.testing import assert_equal


class TestNGramFeaturizer(unittest.TestCase):

    @unittest.skipIf(os.name != "nt" and six.PY2,
                     "encoding/decoding issues with linux py2.7, bug 286536")
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
        X_train = texttransform.fit_transform(X_train[:100])
        sum = X_train.iloc[:].sum().sum()
        print(sum)
        assert_equal(sum, 30513, "sum of all features is incorrect!")


if __name__ == '__main__':
    unittest.main()
