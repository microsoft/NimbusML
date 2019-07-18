# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import os
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
        pickle_filename = 'nimbusml_model.p'
        with open(pickle_filename, 'wb') as f:
            pickle.dump(model_nimbusml, f)

        with open(pickle_filename, "rb") as f:
            model_nimbusml_pickle = pickle.load(f)

        os.remove(pickle_filename)

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

        os.remove('model.nimbusml.m')

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
        pickle_filename = 'nimbusml_model.p'
        with open(pickle_filename, 'wb') as f:
            pickle.dump(model_nimbusml, f)

        with open(pickle_filename, "rb") as f:
            model_nimbusml_pickle = pickle.load(f)

        os.remove(pickle_filename)

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

        os.remove('model.nimbusml.m')

    def test_pipeline_saves_complete_model_file_when_pickled(self):
        model_nimbusml = Pipeline(
            steps=[
                ('cat',
                 OneHotVectorizer() << categorical_columns),
                ('linear',
                 FastLinearBinaryClassifier(
                     shuffle=False,
                     number_of_threads=1))])

        model_nimbusml.fit(train, label)
        metrics, score = model_nimbusml.test(test, test_label, output_scores=True)

        pickle_filename = 'nimbusml_model.p'

        # Save with pickle
        with open(pickle_filename, 'wb') as f:
            pickle.dump(model_nimbusml, f)

        # Remove the pipeline model from disk so
        # that the unpickled pipeline is forced
        # to get its model from the pickled file.
        os.remove(model_nimbusml.model)

        with open(pickle_filename, "rb") as f:
            model_nimbusml_pickle = pickle.load(f)

        os.remove(pickle_filename)

        metrics_pickle, score_pickle = model_nimbusml_pickle.test(
            test, test_label, output_scores=True)

        assert_almost_equal(score.sum().sum(),
                            score_pickle.sum().sum(),
                            decimal=2)

        assert_almost_equal(metrics.sum().sum(),
                            metrics_pickle.sum().sum(),
                            decimal=2)

    def test_unfitted_pickled_pipeline_can_be_fit(self):
        pipeline = Pipeline(
            steps=[
                ('cat',
                 OneHotVectorizer() << categorical_columns),
                ('linear',
                 FastLinearBinaryClassifier(
                     shuffle=False,
                     number_of_threads=1))])

        pipeline.fit(train, label)
        metrics, score = pipeline.test(test, test_label, output_scores=True)

        # Create a new unfitted pipeline
        pipeline = Pipeline(
            steps=[
                ('cat',
                 OneHotVectorizer() << categorical_columns),
                ('linear',
                 FastLinearBinaryClassifier(
                     shuffle=False,
                     number_of_threads=1))])

        pickle_filename = 'nimbusml_model.p'

        # Save with pickle
        with open(pickle_filename, 'wb') as f:
            pickle.dump(pipeline, f)

        with open(pickle_filename, "rb") as f:
            pipeline_pickle = pickle.load(f)

        os.remove(pickle_filename)

        pipeline_pickle.fit(train, label)
        metrics_pickle, score_pickle = pipeline_pickle.test(
            test, test_label, output_scores=True)

        assert_almost_equal(score.sum().sum(),
                            score_pickle.sum().sum(),
                            decimal=2)

        assert_almost_equal(metrics.sum().sum(),
                            metrics_pickle.sum().sum(),
                            decimal=2)

    def test_unpickled_pipeline_has_feature_contributions(self):
        features = ['age', 'education-num', 'hours-per-week']
        
        model_nimbusml = Pipeline(
            steps=[FastLinearBinaryClassifier(feature=features)])
        model_nimbusml.fit(train, label)
        fc = model_nimbusml.get_feature_contributions(test)

        # Save with pickle
        pickle_filename = 'nimbusml_model.p'
        with open(pickle_filename, 'wb') as f:
            pickle.dump(model_nimbusml, f)
        # Unpickle model
        with open(pickle_filename, "rb") as f:
            model_nimbusml_pickle = pickle.load(f)

        fc_pickle = model_nimbusml_pickle.get_feature_contributions(test)

        assert ['FeatureContributions.' + feature in fc_pickle.columns
                for feature in features]

        assert [fc['FeatureContributions.' + feature].equals(
            fc_pickle['FeatureContributions.' + feature])
                for feature in features]

        os.remove(pickle_filename)

    def test_unpickled_predictor_has_feature_contributions(self):
        features = ['age', 'education-num', 'hours-per-week']
        
        model_nimbusml = FastLinearBinaryClassifier(feature=features)
        model_nimbusml.fit(train, label)
        fc = model_nimbusml.get_feature_contributions(test)

        # Save with pickle
        pickle_filename = 'nimbusml_model.p'
        with open(pickle_filename, 'wb') as f:
            pickle.dump(model_nimbusml, f)
        # Unpickle model
        with open(pickle_filename, "rb") as f:
            model_nimbusml_pickle = pickle.load(f)

        fc_pickle = model_nimbusml_pickle.get_feature_contributions(test)

        assert ['FeatureContributions.' + feature in fc_pickle.columns
                for feature in features]

        assert [fc['FeatureContributions.' + feature].equals(
            fc_pickle['FeatureContributions.' + feature])
                for feature in features]

        os.remove(pickle_filename)

    def test_pipeline_loaded_from_zip_has_feature_contributions(self):
        features = ['age', 'education-num', 'hours-per-week']
        
        model_nimbusml = Pipeline(
            steps=[FastLinearBinaryClassifier(feature=features)])
        model_nimbusml.fit(train, label)
        fc = model_nimbusml.get_feature_contributions(test)

        # Save the model to zip
        model_filename = 'nimbusml_model.zip'
        model_nimbusml.save_model(model_filename)
        # Load the model from zip
        model_nimbusml_zip = Pipeline()
        model_nimbusml_zip.load_model(model_filename)

        fc_zip = model_nimbusml_zip.get_feature_contributions(test)
        
        assert ['FeatureContributions.' + feature in fc_zip.columns
                for feature in features]

        assert [fc['FeatureContributions.' + feature].equals(
            fc_zip['FeatureContributions.' + feature])
                for feature in features]

        os.remove(model_filename)

    def test_predictor_loaded_from_zip_has_feature_contributions(self):
        features = ['age', 'education-num', 'hours-per-week']
        
        model_nimbusml = FastLinearBinaryClassifier(feature=features)
        model_nimbusml.fit(train, label)
        fc = model_nimbusml.get_feature_contributions(test)

        # Save the model to zip
        model_filename = 'nimbusml_model.zip'
        model_nimbusml.save_model(model_filename)
        # Load the model from zip
        model_nimbusml_zip = Pipeline()
        model_nimbusml_zip.load_model(model_filename)

        fc_zip = model_nimbusml_zip.get_feature_contributions(test)
        
        assert ['FeatureContributions.' + feature in fc_zip.columns
                for feature in features]

        assert [fc['FeatureContributions.' + feature].equals(
            fc_zip['FeatureContributions.' + feature])
                for feature in features]

        os.remove(model_filename)

if __name__ == '__main__':
    unittest.main()
