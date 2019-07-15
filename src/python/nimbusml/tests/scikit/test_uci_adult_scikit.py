# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import os
import pickle
import unittest

import numpy as np
import pandas as pd
from nimbusml import Pipeline as nimbusmlPipeline
from nimbusml.datasets import get_dataset
from nimbusml.ensemble import FastTreesBinaryClassifier
from nimbusml.feature_extraction.categorical import OneHotVectorizer
from nimbusml.linear_model import FastLinearBinaryClassifier
from nimbusml.linear_model import LogisticRegressionClassifier
from nimbusml.preprocessing.normalization import MeanVarianceScaler
from nimbusml.utils import check_accuracy_scikit, get_X_y
from sklearn.base import clone
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.testing import assert_equal

try:
    from pandas.testing import assert_frame_equal
except ImportError:
    # earlier versions
    from pandas.util.testing import assert_frame_equal

train_file = get_dataset("uciadult_train").as_filepath()
test_file = get_dataset("uciadult_test").as_filepath()
label_column = 'label'
categorical_columns = [
    'workclass',
    'education',
    'marital-status',
    'occupation',
    'relationship',
    'ethnicity',
    'sex',
    'native-country-region']
selected_features = ['age', 'education-num']


class TestUciAdultScikit(unittest.TestCase):

    def test_linear(self):
        np.random.seed(0)
        (train, label) = get_X_y(train_file, label_column, sep=',')
        (test, label1) = get_X_y(test_file, label_column, sep=',')
        pipe = Pipeline(
            steps=[
                ('cat',
                 OneHotVectorizer() << categorical_columns),
                ('linear',
                 FastLinearBinaryClassifier(
                     shuffle=False,
                     number_of_threads=1))])
        pipe.fit(train, label)
        out_data = pipe.predict(test)
        check_accuracy_scikit(test_file, label_column, out_data, 0.779)

    def test_trees(self):
        np.random.seed(0)
        (train, label) = get_X_y(train_file, label_column, sep=',')
        (test, label1) = get_X_y(test_file, label_column, sep=',')
        pipe = Pipeline(
            steps=[
                ('cat',
                 OneHotVectorizer() << categorical_columns),
                ('linear',
                 FastTreesBinaryClassifier())])
        pipe.fit(train, label)
        out_data = pipe.predict(test)
        check_accuracy_scikit(test_file, label_column, out_data, 0.77)

    def test_feature_union(self):
        np.random.seed(0)
        (train, label) = get_X_y(train_file, label_column,
                                 sep=',', features=selected_features)
        (test, label1) = get_X_y(test_file, label_column,
                                 sep=',', features=selected_features)
        fu = FeatureUnion(transformer_list=[
            ('onehot', OneHotEncoder()),
            ('cat', OneHotVectorizer())
        ])
        pipe = Pipeline(
            steps=[
                ('fu', fu), ('linear', FastLinearBinaryClassifier(
                    shuffle=False, number_of_threads=1))])
        pipe.fit(train, label)
        out_data = pipe.predict(test)
        check_accuracy_scikit(test_file, label_column, out_data, 0.709)

    def test_pickle_predictor(self):
        np.random.seed(0)
        (X_train, y_train) = get_X_y(train_file,
                                     label_column, sep=',',
                                     features=selected_features)
        (X_test, y_test) = get_X_y(test_file,
                                   label_column, sep=',',
                                   features=selected_features)

        ftree = FastTreesBinaryClassifier().fit(X_train, y_train)
        scores = ftree.predict(X_test)
        accu1 = np.mean(y_test.values.ravel() == scores.values)

        # Unpickle model and score. We should get the exact same accuracy as
        # above
        s = pickle.dumps(ftree)
        os.remove(ftree.model_)
        ftree2 = pickle.loads(s)
        scores2 = ftree2.predict(X_test)
        accu2 = np.mean(y_test.values.ravel() == scores2.values)
        assert_equal(
            accu1,
            accu2,
            "accuracy mismatch after unpickling predictor")

    def test_pickle_transform(self):
        np.random.seed(0)
        (X_train, y_train) = get_X_y(train_file,
                                     label_column, sep=',',
                                     features=selected_features)
        cat = (OneHotVectorizer() << ['age']).fit(X_train, verbose=0)
        out1 = cat.transform(X_train)

        # Unpickle transform and generate output.
        # We should get the exact same output as above
        s = pickle.dumps(cat)
        os.remove(cat.model_)
        cat2 = pickle.loads(s)
        out2 = cat2.transform(X_train)
        assert_equal(
            out1.sum().sum(),
            out2.sum().sum(),
            "data mismatch after unpickling transform")

    def test_pickle_pipeline(self):
        np.random.seed(0)
        (X_train, y_train) = get_X_y(train_file,
                                     label_column, sep=',',
                                     features=selected_features)
        (X_test, y_test) = get_X_y(test_file,
                                   label_column, sep=',',
                                   features=selected_features)
        if 'F1' in X_train.columns:
            raise Exception("F1 is in the dataset")
        cat = OneHotVectorizer() << 'age'
        ftree = FastTreesBinaryClassifier()
        pipe = Pipeline(steps=[("cat", cat), ("ftree", ftree)])
        pipe.fit(X_train, y_train)

        scores = pipe.predict(X_test)
        accu1 = np.mean(y_test.values.ravel() == scores.values)

        # Unpickle model and score. We should get the exact same accuracy as
        # above
        s = pickle.dumps(pipe)
        os.remove(cat.model_)
        os.remove(ftree.model_)
        pipe2 = pickle.loads(s)

        scores2 = pipe2.predict(X_test)
        accu2 = np.mean(y_test.values.ravel() == scores2.values)
        assert_equal(
            accu1,
            accu2,
            "accuracy mismatch after unpickling pipeline")

    def test_pickle_pipeline_unnamed(self):
        (X_train, y_train) = get_X_y(train_file,
                                     label_column, sep=',',
                                     features=selected_features)
        (X_test, y_test) = get_X_y(test_file,
                                   label_column, sep=',',
                                   features=selected_features)
        if 'F1' in X_train.columns:
            raise Exception("F1 is in the dataset")
        cat = OneHotVectorizer() << 'age'
        ftree = FastTreesBinaryClassifier()
        pipe = nimbusmlPipeline([cat, ftree])
        pipe.fit(X_train, y_train, verbose=0)

        scores = pipe.predict(X_test)
        accu1 = np.mean(y_test.values.ravel() == scores["PredictedLabel"].values)

        # Unpickle model and score. We should get the exact same accuracy as
        # above
        s = pickle.dumps(pipe)
        pipe2 = pickle.loads(s)
        scores2 = pipe2.predict(X_test)
        accu2 = np.mean(y_test.values.ravel() == scores2["PredictedLabel"].values)
        assert_equal(
            accu1,
            accu2,
            "accuracy mismatch after unpickling pipeline")
        assert_frame_equal(scores, scores2)

    def test_pickle_pipeline_and_nimbusml_pipeline(self):
        (X_train, y_train) = get_X_y(train_file,
                                     label_column, sep=',',
                                     features=selected_features)
        (X_test, y_test) = get_X_y(test_file,
                                   label_column, sep=',',
                                   features=selected_features)
        if 'F1' in X_train.columns:
            raise Exception("F1 is in the dataset")
        cat = OneHotVectorizer() << 'age'
        ftree = FastTreesBinaryClassifier()
        nimbusmlpipe = nimbusmlPipeline([cat, ftree])
        skpipe = Pipeline(steps=[('nimbusml', nimbusmlpipe)])
        skpipe.fit(X_train, y_train)

        scores = skpipe.predict(X_test)
        accu1 = np.mean(y_test.values.ravel() == scores["PredictedLabel"].values)

        # Unpickle model and score. We should get the exact same accuracy as
        # above
        s = pickle.dumps(skpipe)
        pipe2 = pickle.loads(s)
        scores2 = pipe2.predict(X_test)
        accu2 = np.mean(y_test.values.ravel() == scores2["PredictedLabel"].values)
        assert_equal(
            accu1,
            accu2,
            "accuracy mismatch after unpickling pipeline")
        assert_frame_equal(scores, scores2)

    def test_pipeline_clone(self):
        (X_train, y_train) = get_X_y(train_file,
                                     label_column, sep=',',
                                     features=selected_features)
        (X_test, y_test) = get_X_y(test_file,
                                   label_column, sep=',',
                                   features=selected_features)
        if 'F1' in X_train.columns:
            raise Exception("F1 is in the dataset")
        cat = OneHotVectorizer() << 'age'
        ftree = FastTreesBinaryClassifier()
        nimbusmlpipe = nimbusmlPipeline([cat, ftree])
        skpipe = Pipeline(steps=[('nimbusml', nimbusmlpipe)])
        skpipe.fit(X_train, y_train)

        scores = skpipe.predict(X_test)

        copy = clone(skpipe)
        scores2 = copy.predict(X_test)
        assert_frame_equal(scores, scores2)

        # checks we can fit again
        skpipe.fit(X_train, y_train)
        scores3 = skpipe.predict(X_test)
        assert_frame_equal(scores, scores3)

    def test_pipeline_get_params(self):

        (X_train, y_train) = get_X_y(train_file,
                                     label_column, sep=',',
                                     features=selected_features)
        (X_test, y_test) = get_X_y(test_file,
                                   label_column, sep=',',
                                   features=selected_features)
        if 'F1' in X_train.columns:
            raise Exception("F1 is in the dataset")
        cat = OneHotVectorizer() << 'age'
        ftree = FastTreesBinaryClassifier()
        nimbusmlpipe = nimbusmlPipeline([cat, ftree])
        skpipe = Pipeline(steps=[('nimbusml', nimbusmlpipe)])
        skpipe.fit(X_train, y_train)
        pars = skpipe.get_params(deep=True)
        assert 'steps' in pars
        step = pars['steps'][0]
        assert len(step) == 2
        assert 'nimbusml' in pars
        assert 'nimbusml__random_state' in pars
        assert 'nimbusml__steps' in pars

    def test_pipeline_grid_search(self):
        (X_train, y_train) = get_X_y(train_file,
                                     label_column, sep=',',
                                     features=selected_features)
        (X_test, y_test) = get_X_y(test_file,
                                   label_column, sep=',',
                                   features=selected_features)
        if 'F1' in X_train.columns:
            raise Exception("F1 is in the dataset")
        cat = OneHotVectorizer() << 'age'
        ftree = FastTreesBinaryClassifier(number_of_trees=5)
        pipe = Pipeline(
            steps=[
                ("cat", cat), ('pca', PCA(5)), ("ftree", ftree)])

        grid = GridSearchCV(pipe, dict(pca__n_components=[2],
                                       ftree__number_of_trees=[11]))
        grid.fit(X_train, y_train)
        assert grid.best_params_ == {
            'ftree__number_of_trees': 11,
            'pca__n_components': 2}
        steps = grid.best_estimator_.steps
        ft = steps[-1][1]
        number_of_trees = ft.number_of_trees
        assert number_of_trees == 11

    def test_lr_named_steps_iris(self):
        iris = load_iris()
        X = iris.data[:, :2]  # we only take the first two features.
        y = iris.target
        df = pd.DataFrame(X, columns=['X1', 'X2'])
        df['Label'] = y
        pipe = nimbusmlPipeline([('norm', MeanVarianceScaler() << ['X1', 'X2']),
                            ('lr',
                             LogisticRegressionClassifier() << ['X1', 'X2'])])
        pipe.fit(df)
        pred = pipe.predict(df).head()
        assert len(pred) == 5


if __name__ == '__main__':
    unittest.main()
