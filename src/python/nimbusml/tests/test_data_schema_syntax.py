# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
import os
import tempfile
import unittest

import numpy
import pandas
import sklearn.linear_model
from nimbusml import Pipeline
from nimbusml.feature_extraction.categorical import OneHotVectorizer
from nimbusml.feature_extraction.text import NGramFeaturizer
from nimbusml.internal.entrypoints._ngramextractor_ngram import n_gram
from nimbusml.internal.utils.data_stream import FileDataStream
from nimbusml.linear_model import FastLinearBinaryClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline as ppl


class TestDataSchemaSyntax(unittest.TestCase):

    def _test_schema_syntax_shift(self):
        df = pandas.DataFrame(
            data=dict(
                X1=[
                    0.1, 0.2], X2=[
                    0.1, 0.2], yl=[
                    1, 0], tx=[
                    'e', 'r'], txn=[
                    'e', 'r']))

        exp = Pipeline([OneHotVectorizer() << 'tx',
                        OneHotVectorizer() << {'tx3': 'tx'},
                        FastLinearBinaryClassifier()])
        exp.fit(df, 'yl')

    def _test_schema_syntax_shift_file(self):
        df = pandas.DataFrame(data=dict(X1=[0.1, 0.2], X2=[0.1, 0.2],
                                        yl=[1.0, 0.0], tx=['e', 'r']))

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            df.to_csv(f, sep=',', index=False)

        fi = FileDataStream.read_csv(
            f.name, sep=',', dtype={
                'yl': numpy.float32})
        assert str(
            fi.schema) == 'col=X1:R8:0 col=X2:R8:1 col=tx:TX:2 col=yl:R4:3 \
            "header=+ sep=,'

        exp = Pipeline([OneHotVectorizer() << fi['tx'],
                        FastLinearBinaryClassifier()])
        exp.fit(df, 'yl')

        os.remove(f.name)

    def _test_schema_syntax_shift_df(self):
        df = pandas.DataFrame(data=dict(X1=[0.1, 0.2], X2=[0.1, 0.2],
                                        yl=[1, 0], tx=['e', 'r']))

        exp = Pipeline([OneHotVectorizer() << 'tx',
                        FastLinearBinaryClassifier()])
        exp.fit(df, 'yl')

    def _test_sklearn_pipeline(self):
        train_reviews = pandas.DataFrame(data=dict(
            review=["This is great", "I hate it", "Love it", "Do not like it"],
            like=[True, False, True, False]))
        y = train_reviews['like']
        int_y = [int(x) for x in y]
        X = train_reviews.loc[:, train_reviews.columns != 'like']
        featurizer = NGramFeaturizer(word_feature_extractor=n_gram())
        svd = TruncatedSVD(random_state=1, n_components=5)
        lr = sklearn.linear_model.LogisticRegression()
        pipe1 = ppl([("featurizer", featurizer), ("svd", svd), ("lr", lr)])
        pipe1.fit(X, int_y)
        pred = pipe1.predict(X)
        assert pred.shape == (4,)

    def test_schema_syntax_multilevel(self):
        df = pandas.DataFrame(data=dict(X1=[0.1, 0.2], X2=[0.1, 0.2],
                                        yl=[1, 0], tx=['e', 'r']))
        columns = [('X', 'X1'), ('X', 'X2'), ('Y', 'yl'), ('TX', 'tx')]
        df.columns = pandas.MultiIndex.from_tuples(columns)

        exp = Pipeline([OneHotVectorizer() << ('TX', 'tx'),
                        FastLinearBinaryClassifier()])

        assert exp.nodes[0]._columns == ('TX', 'tx')
        assert exp.nodes[0].input == [('TX', 'tx')]
        exp.fit(df, ('Y', 'yl'))
        pred = exp.predict(df)
        assert pred.shape == (2, 3)


if __name__ == "__main__":
    unittest.main()
