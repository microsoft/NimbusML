# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import unittest

import pandas
from nimbusml import Pipeline
from nimbusml.feature_extraction.text import LightLda
from nimbusml.feature_extraction.text import NGramFeaturizer
from nimbusml.internal.entrypoints._ngramextractor_ngram import n_gram
from sklearn.utils.testing import assert_almost_equal


class TestIIDLightLda(unittest.TestCase):

    @unittest.skip(
        "Skip temporarily. Will work once LDANative.dll is packaged in pub")
    def test_LightLda(self):
        topics = pandas.DataFrame(data=dict(review=[
            "animals birds cats dogs fish horse",
            "horse birds house fish duck cats",
            "car truck driver bus pickup",
            "car truck driver bus pickup horse ",
            "car truck",
            "bus pickup",
            "space galaxy universe radiation",
            "radiation galaxy universe duck"]))

        pipeline = Pipeline([NGramFeaturizer(word_feature_extractor=n_gram(
        ), vector_normalizer='None') << 'review', LightLda(num_topic=3)])
        y = pipeline.fit_transform(topics)
        assert_almost_equal(
            y.sum().sum(),
            7.000000044,
            decimal=8,
            err_msg="Sum should be %s" %
                    7.000000044)


if __name__ == '__main__':
    unittest.main()
