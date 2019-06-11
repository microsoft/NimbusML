# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import platform
import unittest

import numpy as np
import six
from nimbusml.datasets import get_dataset
from nimbusml.ensemble import LightGbmClassifier
from nimbusml.feature_extraction.text import NGramFeaturizer
from nimbusml.internal.entrypoints._ngramextractor_ngram import n_gram
from nimbusml.utils import get_X_y
from sklearn.model_selection import train_test_split
from sklearn.utils.testing import assert_greater


class TestLightGbmClassifier(unittest.TestCase):

    @unittest.skipIf(platform.system() in ("Linux", "Darwin") and six.PY2,
                     "encoding/decoding issues with linux py2.7, bug 286536")
    def test_lightgbmclassifier(self):
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
        X_train = texttransform.fit_transform(X_train, max_slots=5000)
        X_test = texttransform.transform(X_test, max_slots=5000)

        mymodel = LightGbmClassifier().fit(X_train, y_train, verbose=0)
        scores = mymodel.predict(X_test)
        accuracy = np.mean(y_test.values.ravel() == scores.values)
        assert_greater(
            accuracy,
            0.58,
            "accuracy should be greater than %s" %
            0.58)


if __name__ == '__main__':
    unittest.main()
