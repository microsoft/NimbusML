# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
import unittest

import pandas
from nimbusml import FileDataStream
from nimbusml.datasets import get_dataset
from nimbusml.feature_extraction.text import NGramFeaturizer
from nimbusml.internal.entrypoints._ngramextractor_ngram import n_gram
from nimbusml.linear_model import LogisticRegressionBinaryClassifier


class TestNGramFeaturizer(unittest.TestCase):

    def test_ngramfeaturizer(self):

        train_reviews = pandas.DataFrame(
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

        test_reviews = pandas.DataFrame(
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

        y = train_reviews['like']
        X = train_reviews.loc[:, train_reviews.columns != 'like']

        textt = NGramFeaturizer(word_feature_extractor=n_gram()) << 'review'
        X = textt.fit_transform(X)

        assert X.shape == (25, 116)

        mymodel = LogisticRegressionBinaryClassifier().fit(X, y, verbose=0)
        X_test = textt.transform(test_reviews)
        scores = mymodel.predict(textt.transform(test_reviews))

        # View the scores
        assert scores.shape == (10,)
        assert X_test.shape[0] == 10

    def test_ngramfeaturizer_syntax_dict(self):

        train_reviews = pandas.DataFrame(
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

        test_reviews = pandas.DataFrame(
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

        y = train_reviews['like']
        X = train_reviews.loc[:, train_reviews.columns != 'like']

        textt = NGramFeaturizer(
            word_feature_extractor=n_gram()) << {
            'outg': ['review']}
        X = textt.fit_transform(X)

        assert X.shape == (25, 117)
        # columns ordering changed between 0.22 and 0.23
        assert 'review' in (X.columns[0], X.columns[-1])
        X = X.drop('review', axis=1)

        mymodel = LogisticRegressionBinaryClassifier().fit(X, y, verbose=0)
        X_test = textt.transform(test_reviews)
        X_test = X_test.drop('review', axis=1)
        scores = mymodel.predict(X_test)

        # View the scores
        assert scores.shape == (10,)

    def test_ngramfeaturizer_single(self):

        path = get_dataset('infert').as_filepath()
        file_schema = 'sep=, col=id:TX:0 col=education:TX:1 col=age:R4:2 ' \
                      'col=parity:R4:3 col=induced:R4:4 col=case:R4:5 ' \
                      'col=spontaneous:R4:6 quote+ header=+'
        data = FileDataStream(path, schema=file_schema)
        xf = NGramFeaturizer(word_feature_extractor=n_gram(),
                             columns={'features': ['id', 'education']})

        features = xf.fit_transform(data)
        assert features.shape == (248, 652)

    def test_ngramfeaturizer_multi(self):

        path = get_dataset('infert').as_filepath()
        file_schema = 'sep=, col=id:TX:0 col=education:TX:1 col=age:R4:2 ' \
                      'col=parity:R4:3 col=induced:R4:4 col=case:R4:5 ' \
                      'col=spontaneous:R4:6 quote+ header=+'
        data = FileDataStream(path, schema=file_schema)
        try:
            xf = NGramFeaturizer(word_feature_extractor=n_gram(),
                                 columns={'features': ['id'],
                                          'features2': ['education']})
        except TypeError as e:
            assert 'Only one output column is allowed' in str(e)
            return

        try:
            # System.InvalidCastException: 'Cannot cast
            # Newtonsoft.Json.Linq.JArray to Newtonsoft.Json.Linq.JToken.
            xf.fit_transform(data)
            assert False
        except RuntimeError:
            pass


if __name__ == '__main__':
    unittest.main()
