# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
import unittest

import pandas
# need to be used later with new syntax
from nimbusml import Pipeline
from nimbusml.ensemble import LightGbmClassifier
from nimbusml.feature_extraction.categorical import OneHotVectorizer
from nimbusml.feature_extraction.text import NGramFeaturizer
from nimbusml.internal.entrypoints._ngramextractor_ngram import n_gram
from nimbusml.linear_model import FastLinearBinaryClassifier


class TestSyntaxOneHotVectorizer(unittest.TestCase):

    def get_simple_df(self):
        df = pandas.DataFrame(dict(education=['A', 'B', 'A', 'B', 'A'],
                                   education2=['c', 'd', 'c', 'd', 'c'],
                                   workclass=['X', 'X', 'Y', 'Y', 'Y'],
                                   y=[1, 0, 1, 0, 0]))
        X = df.drop('y', axis=1)
        y = df['y']
        return df, X, y

    def test_syntax0_passing(self):
        df, X, y = self.get_simple_df()
        exp = Pipeline([
            OneHotVectorizer() << {'f1': 'education2'},
            OneHotVectorizer(max_num_terms=2) << {'f3': 'workclass'},
            FastLinearBinaryClassifier() << ['f1', 'f3']
        ])
        exp.fit(X, y)
        res = exp.transform(X)
        assert res.shape == (5, 16)

    def test_syntax1_passing(self):
        df, X, y = self.get_simple_df()
        exp = Pipeline([
            OneHotVectorizer() << {'f1': 'education2'},
            OneHotVectorizer(max_num_terms=2) << {'f3': 'workclass'},
            LightGbmClassifier(minimum_example_count_per_leaf=1) << ['f1', 'f3']
        ])
        exp.fit(X, y)
        res = exp.transform(X)
        assert res.shape == (5, 16)

    def test_syntax2_passing(self):
        df, X, y = self.get_simple_df()
        exp = Pipeline([
            OneHotVectorizer() << {'f1': ['education']},
            OneHotVectorizer(max_num_terms=2) << {'f3': 'workclass'},
            FastLinearBinaryClassifier() << ['f1', 'f3']
        ])
        exp.fit(X, y)
        res = exp.transform(X)
        assert res.shape == (5, 16)

    def test_syntax3_passing(self):
        df, X, y = self.get_simple_df()
        vec = OneHotVectorizer() << ['education', 'education2']
        vec.fit(X)
        res = vec.transform(X)
        assert res.shape == (5, 5)

    def test_syntax4_passing(self):
        df, X, y = self.get_simple_df()
        vec = OneHotVectorizer() << {'edu1': ['education']}
        vec.fit(X)
        res = vec.transform(X)
        assert res.shape == (5, 5)

    def test_syntax5_failing(self):
        df, X, y = self.get_simple_df()
        vec = OneHotVectorizer() << {'edu1': ['education1']}
        try:
            vec.fit_transform(X, verbose=2)
            assert False
        except RuntimeError as e:
            assert "Returned code is -1. Check the log for error messages.." \
                   in str(e)
        vec = OneHotVectorizer() << {'edu1': ['education']}
        res = vec.fit_transform(X)
        assert res.shape == (5, 5)

    def test_syntax6_passing(self):
        df, X, y = self.get_simple_df()
        vec = OneHotVectorizer() << ['education', 'education2']
        res = vec.fit_transform(X)
        assert res.shape == (5, 5)

    def test_syntax7_columes(self):
        df = pandas.DataFrame(dict(education1=['A', 'B', 'A', 'B', 'A'],
                                   education2=['c', 'd', 'c', 'd', 'c'],
                                   workclass=['X', 'X', 'Y', 'Y', 'Y'],
                                   y=[1, 0, 1, 0, 0]))

        X = df.drop('y', axis=1)
        y = df['y']
        try:
            OneHotVectorizer(columes={
                'edu1': ['education1', 'education2']
            }).fit(X, y)
            raise AssertionError("columnes should have been caught")
        except NameError as e:
            if "Parameter 'columes' is not allowed for class " \
               "'OneHotVectorizer'" not in str(e):
                raise
        try:
            OneHotVectorizer(columees={'edu1': ['education1', 'education2']})
            raise AssertionError("columnes should have been caught")
        except NameError as e:
            if "Parameter 'columees' is not allowed for class " \
               "'OneHotVectorizer'" not in str(e):
                raise

    def test_syntax8_simple_input(self):
        df = pandas.DataFrame(dict(education1=['A', 'B', 'A', 'B', 'A'],
                                   education2=['c', 'd', 'c', 'd', 'c'],
                                   workclass=['X', 'X', 'Y', 'Y', 'Y'],
                                   y=[1, 0, 1, 0, 0]))
        X = df.drop('y', axis=1)
        y = df['y']
        # Bug : Works with incorrect spelling of 'columes'.
        # But it should not accept multiple inputs since its a Categorical
        # transform
        try:
            OneHotVectorizer(columns={
                'edu1': ['education1', 'education2']
            }).fit(X, y)
            raise AssertionError("multiple allowed forbidden")
        except RuntimeError as e:
            if "Number of inputs and outputs should be equal " \
               "for type" not in str(e):
                raise

    def test_syntax9_multiple_inputs(self):
        df = pandas.DataFrame(dict(education1=['A', 'B', 'A', 'B', 'A'],
                                   education2=['c', 'd', 'c', 'd', 'c'],
                                   workclass=['X', 'X', 'Y', 'Y', 'Y'],
                                   y=[1, 0, 1, 0, 0]))
        X = df.drop('y', axis=1)

        ng4 = NGramFeaturizer(word_feature_extractor=n_gram()) << {
            'out1': ['education1', 'education2']}
        output4 = ng4.fit_transform(X)
        assert output4.shape == (5, 13)
