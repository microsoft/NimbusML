# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import pickle
import unittest

from nimbusml import Pipeline
from nimbusml.datasets import get_dataset
from nimbusml.feature_extraction.categorical import OneHotVectorizer
from nimbusml.linear_model import FastLinearBinaryClassifier
from nimbusml.utils import get_X_y
from numpy.testing import assert_almost_equal

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
(train, label) = get_X_y(train_file, label_column, sep=',')
(test, test_label) = get_X_y(test_file, label_column, sep=',')


class TestLoadSave(unittest.TestCase):

    def test_model_dataframe(self):
        model_nimbusml = Pipeline(
            steps=[
                ('cat',
                 OneHotVectorizer() << categorical_columns),
                ('linear',
                 FastLinearBinaryClassifier(
                     shuffle=False,
                     number_of_threads=1))])

        model_nimbusml.fit(train, label)

        # Save with pickle
        pickle.dump(model_nimbusml, open('nimbusml_model.p', 'wb'))
        model_nimbusml_pickle = pickle.load(open("nimbusml_model.p", "rb"))

        score1 = model_nimbusml.predict(test).head(5)
        score2 = model_nimbusml_pickle.predict(test).head(5)

        metrics, score = model_nimbusml.test(test, test_label, output_scores=True)
        metrics_pickle, score_pickle = model_nimbusml_pickle.test(
            test, test_label, output_scores=True)

        # Save load with pipeline methods
        model_nimbusml.save_model('model.nimbusml.m')
        model_nimbusml_load = Pipeline()
        model_nimbusml_load.load_model('model.nimbusml.m')

        score1 = model_nimbusml.predict(test).head(5)
        score2 = model_nimbusml_load.predict(test).head(5)

        metrics, score = model_nimbusml.test(test, test_label, output_scores=True)
        model_nimbusml_load, score_load = model_nimbusml_load.test(
            test, test_label, evaltype='binary', output_scores=True)

        assert_almost_equal(score1.sum().sum(), score2.sum().sum(), decimal=2)
        assert_almost_equal(
            metrics.sum().sum(),
            model_nimbusml_load.sum().sum(),
            decimal=2)

    def test_model_datastream(self):
        model_nimbusml = Pipeline(
            steps=[
                ('cat',
                 OneHotVectorizer() << categorical_columns),
                ('linear',
                 FastLinearBinaryClassifier(
                     shuffle=False,
                     number_of_threads=1))])

        model_nimbusml.fit(train, label)

        # Save with pickle
        pickle.dump(model_nimbusml, open('nimbusml_model.p', 'wb'))
        model_nimbusml_pickle = pickle.load(open("nimbusml_model.p", "rb"))

        score1 = model_nimbusml.predict(test).head(5)
        score2 = model_nimbusml_pickle.predict(test).head(5)

        metrics, score = model_nimbusml.test(test, test_label, output_scores=True)
        metrics_pickle, score_pickle = model_nimbusml_pickle.test(
            test, test_label, output_scores=True)

        assert_almost_equal(score1.sum().sum(), score2.sum().sum(), decimal=2)
        assert_almost_equal(
            metrics.sum().sum(),
            metrics_pickle.sum().sum(),
            decimal=2)

        # Save load with pipeline methods
        model_nimbusml.save_model('model.nimbusml.m')
        model_nimbusml_load = Pipeline()
        model_nimbusml_load.load_model('model.nimbusml.m')

        score1 = model_nimbusml.predict(test).head(5)
        score2 = model_nimbusml_load.predict(test).head(5)

        metrics, score = model_nimbusml.test(test, test_label, output_scores=True)
        model_nimbusml_load, score_load = model_nimbusml_load.test(
            test, test_label, evaltype='binary', output_scores=True)

        assert_almost_equal(score1.sum().sum(), score2.sum().sum(), decimal=2)
        assert_almost_equal(
            metrics.sum().sum(),
            model_nimbusml_load.sum().sum(),
            decimal=2)


if __name__ == '__main__':
    unittest.main()
