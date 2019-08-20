# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import os
import platform
import unittest

import numpy as np
import pandas as pd
import six
from nimbusml import Pipeline
from nimbusml.datasets import get_dataset
from nimbusml.ensemble import FastTreesBinaryClassifier
from nimbusml.feature_extraction.categorical import OneHotHashVectorizer, \
    OneHotVectorizer
from nimbusml.feature_extraction.text import NGramFeaturizer
from nimbusml.feature_extraction.text import WordEmbedding
from nimbusml.feature_extraction.text.extractor import Ngram
from nimbusml.linear_model import FastLinearBinaryClassifier
from nimbusml.utils import get_X_y
from sklearn.model_selection import GridSearchCV
from sklearn.utils.testing import assert_raises

train_file = get_dataset('uciadult_train').as_filepath()
test_file = get_dataset('uciadult_test').as_filepath()
categorical_columns = [
    'workclass',
    'education',
    'marital-status',
    'occupation',
    'relationship',
    'ethnicity',
    'sex',
    'native-country-region']
label_column = 'label'


class TestSweep(unittest.TestCase):

    def test_hyperparameters_sweep(self):
        # general test with combination of named and unnamed steps
        np.random.seed(0)
        df = pd.DataFrame(dict(education=['A', 'A', 'A', 'A', 'B', 'A', 'B'],
                               workclass=['X', 'Y', 'X', 'X', 'X', 'Y', 'Y'],
                               y=[1, 0, 1, 1, 0, 1, 0]))
        X = df.drop('y', axis=1)
        y = df['y']
        pipe = Pipeline([
            ('cat', OneHotVectorizer() << 'education'),
            # unnamed step, stays same in grid search
            OneHotHashVectorizer() << 'workclass',
            # number_of_trees 0 will actually be never run by grid search
            ('learner', FastTreesBinaryClassifier(number_of_trees=0, number_of_leaves=2))
        ])

        param_grid = dict(
            cat__output_kind=[
                'Indicator', 'Binary'], learner__number_of_trees=[
                1, 2, 3])
        grid = GridSearchCV(pipe, param_grid)

        grid.fit(X, y)
        print(grid.best_params_)
        assert grid.best_params_ == {
            'cat__output_kind': 'Indicator',
            'learner__number_of_trees': 1}

    def test_learners_sweep(self):
        # grid search over 2 learners, even though pipe defined with
        # FastTreesBinaryClassifier
        # FastLinearBinaryClassifier learner wins, meaning we grid searched
        # over it
        np.random.seed(0)

        df = pd.DataFrame(dict(education=['A', 'A', 'A', 'A', 'B', 'A', 'B'],
                               workclass=['X', 'Y', 'X', 'X', 'X', 'Y', 'Y'],
                               y=[1, 0, 1, 1, 0, 1, 0]))
        X = df.drop('y', axis=1)
        y = df['y']

        cat = OneHotVectorizer() << ['education', 'workclass']
        learner = FastTreesBinaryClassifier()
        pipe = Pipeline(steps=[('cat', cat), ('learner', learner)])

        param_grid = dict(
            learner=[
                FastLinearBinaryClassifier(),
                FastTreesBinaryClassifier()],
            learner__number_of_threads=[
                1,
                4])
        grid = GridSearchCV(pipe, param_grid)

        grid.fit(X, y)
        assert grid.best_params_[
            'learner'].__class__.__name__ == 'FastLinearBinaryClassifier'
        assert grid.best_params_['learner__number_of_threads'] == 1

    @unittest.skipIf(
        six.PY2,
        "potential bug in pandas read_csv of unicode text in python2.7")
    def test_uciadult_sweep(self):
        # grid search over number_of_trees and then confirm the best number_of_trees by
        # full train
        np.random.seed(0)
        (X_train, y_train) = get_X_y(train_file,
                                     label_column, sep=',', encoding='utf-8')
        (X_test, y_test) = get_X_y(test_file,
                                   label_column, sep=',', encoding='utf-8')

        cat = OneHotHashVectorizer() << categorical_columns
        # number_of_trees 100 will actually be never run by grid search
        # as its not in param_grid below
        learner = FastTreesBinaryClassifier(number_of_trees=100, number_of_leaves=5)
        pipe = Pipeline(steps=[('cat', cat), ('learner', learner)])

        param_grid = dict(learner__number_of_trees=[1, 5, 10])
        grid = GridSearchCV(pipe, param_grid)

        grid.fit(X_train, y_train)
        assert grid.best_params_['learner__number_of_trees'] == 10

        # compare AUC on number_of_trees 1, 5, 10
        pipe.set_params(learner__number_of_trees=1)
        pipe.fit(X_train, y_train)
        metrics1, _ = pipe.test(X_train, y_train)

        pipe.set_params(learner__number_of_trees=5)
        pipe.fit(X_train, y_train)
        metrics5, _ = pipe.test(X_train, y_train)

        pipe.set_params(learner__number_of_trees=10)
        pipe.fit(X_train, y_train)
        metrics10, _ = pipe.test(X_train, y_train)

        assert metrics10['AUC'][0] > metrics5['AUC'][0]
        assert metrics10['AUC'][0] > metrics1['AUC'][0]
        assert metrics10['AUC'][0] > 0.59

    # Problem with the SSL CA cert (path? access rights?) for the build
    # machines to download resources for wordembedding transform
    @unittest.skipIf(
        os.name != "nt" and (
            platform.linux_distribution()[0] != "Ubuntu" or
            platform.linux_distribution()[1] != "16.04"),
        "not supported on this platform")
    def test_NGramFeaturizer_sweep(self):
        # grid search over number_of_trees and then confirm the best number_of_trees by
        # full train
        np.random.seed(0)
        data = pd.DataFrame(
            {
                'review': [
                    'I like this movie',
                    'I don\'t like this',
                    'It is nice',
                    'I like this movie',
                    'I don\'t like this',
                    'It is nice',
                    'So boring'],
                'sentiment': [
                    'pos',
                    'neg',
                    'pos',
                    'pos',
                    'neg',
                    'pos',
                    'neg']})
        pipeline = Pipeline(
            [
                ('ng',
                 NGramFeaturizer(
                     word_feature_extractor=Ngram(),
                     output_tokens_column_name='review_TransformedText',
                     columns='review')),
                WordEmbedding(
                    columns='review_TransformedText',
                    model_kind='SentimentSpecificWordEmbedding'),
                ('lr',
                 FastLinearBinaryClassifier(
                     feature=[
                         'review',
                         'review_TransformedText'],
                     number_of_threads=1,
                     shuffle=False))])

        param_grid = dict(lr__maximum_number_of_iterations=[1, 20])
        grid = GridSearchCV(pipeline, param_grid)

        grid.fit(data['review'], 1 * (data['sentiment'] == 'pos'))
        assert grid.best_params_['lr__maximum_number_of_iterations'] == 20

    # Problem with the SSL CA cert (path? access rights?) for the build
    # machines to download resources for wordembedding transform
    @unittest.skipIf(
        os.name != "nt" and (
            platform.linux_distribution()[0] != "Ubuntu" or
            platform.linux_distribution()[1] != "16.04"),
        "not supported on this platform")
    def test_NGramFeaturizer_glove(self):
        # grid search over number_of_trees and then confirm the best number_of_trees by
        # full train
        np.random.seed(0)
        data = pd.DataFrame(
            {
                'review': [
                    'I like this movie',
                    'I don\'t like this',
                    'It is nice',
                    'I like this movie',
                    'I don\'t like this',
                    'It is nice',
                    'So boring'],
                'sentiment': [
                    'pos',
                    'neg',
                    'pos',
                    'pos',
                    'neg',
                    'pos',
                    'neg']})
        pipeline = Pipeline(
            [
                ('ng',
                 NGramFeaturizer(
                     word_feature_extractor=Ngram(),
                     output_tokens_column_name='review_TransformedText',
                     columns='review')),
                WordEmbedding(
                    columns='review_TransformedText',
                    model_kind='GloVe50D'),
                ('lr',
                 FastLinearBinaryClassifier(
                     feature=[
                         'review',
                         'review_TransformedText'],
                     number_of_threads=1,
                     shuffle=False))])

        param_grid = dict(lr__maximum_number_of_iterations=[1, 100, 20])
        grid = GridSearchCV(pipeline, param_grid)

        grid.fit(data['review'], 1 * (data['sentiment'] == 'pos'))
        assert grid.best_params_['lr__maximum_number_of_iterations'] == 100

    def test_clone_sweep(self):
        # grid search, then clone pipeline and grid search again
        # results should be same
        np.random.seed(0)
        (X_train, y_train) = get_X_y(train_file,
                                     label_column, sep=',', encoding='utf-8')
        (X_test, y_test) = get_X_y(test_file,
                                   label_column, sep=',', encoding='utf-8')

        cat = OneHotHashVectorizer() << categorical_columns
        learner = FastTreesBinaryClassifier(number_of_trees=100, number_of_leaves=5)
        pipe = Pipeline(steps=[('cat', cat), ('learner', learner)])

        param_grid = dict(learner__number_of_trees=[1, 5, 10])
        grid = GridSearchCV(pipe, param_grid)
        grid.fit(X_train, y_train)

        pipe1 = pipe.clone()
        grid1 = GridSearchCV(pipe1, param_grid)
        grid1.fit(X_train, y_train)

        assert grid.best_params_[
            'learner__number_of_trees'] == grid1.best_params_[
            'learner__number_of_trees']

    def test_error_conditions(self):
        # grid search on a wrong param
        np.random.seed(0)
        (X_train, y_train) = get_X_y(train_file,
                                     label_column, sep=',', encoding='utf-8')
        (X_test, y_test) = get_X_y(test_file,
                                   label_column, sep=',', encoding='utf-8')

        cat = OneHotHashVectorizer() << categorical_columns
        learner = FastTreesBinaryClassifier(number_of_trees=100, number_of_leaves=5)
        pipe = Pipeline(steps=[('cat', cat), ('learner', learner)])

        param_grid = dict(learner__wrong_arg=[1, 5, 10])
        grid = GridSearchCV(pipe, param_grid)

        assert_raises(ValueError, grid.fit, X_train, y_train)


if __name__ == '__main__':
    unittest.main()
