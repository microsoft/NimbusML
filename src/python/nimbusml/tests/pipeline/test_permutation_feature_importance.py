# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
import os
import tempfile
import unittest

from nimbusml import FileDataStream
from nimbusml import Pipeline
from nimbusml.datasets import get_dataset
from nimbusml.ensemble import LightGbmRanker
from nimbusml.feature_extraction.categorical import OneHotVectorizer
from nimbusml.linear_model import LogisticRegressionBinaryClassifier, \
    FastLinearClassifier, FastLinearRegressor
from nimbusml.preprocessing import ToKey
from numpy.testing import assert_almost_equal
from pandas.testing import assert_frame_equal


def get_temp_model_file():
    fd, file_name = tempfile.mkstemp(suffix='.zip')
    fl = os.fdopen(fd, 'w')
    fl.close()
    return file_name


class TestPermutationFeatureImportance(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        adult_path = get_dataset('uciadult_train').as_filepath()
        self.classification_data = FileDataStream.read_csv(adult_path)
        binary_pipeline = Pipeline([
            OneHotVectorizer(columns=['education']),
            LogisticRegressionBinaryClassifier(
                feature=['age', 'education'], label='label',
                number_of_threads=1)])
        self.binary_model = binary_pipeline.fit(self.classification_data)
        self.binary_pfi = self.binary_model.permutation_feature_importance(self.classification_data)
        classifier_pipeline = Pipeline([
            OneHotVectorizer(columns=['education']),
            FastLinearClassifier(feature=['age', 'education'], label='label',
                                 number_of_threads=1, shuffle=False)])
        self.classifier_model = classifier_pipeline.fit(self.classification_data)
        self.classifier_pfi = self.classifier_model.permutation_feature_importance(self.classification_data)

        infert_path = get_dataset('infert').as_filepath()
        self.regression_data = FileDataStream.read_csv(infert_path)
        regressor_pipeline = Pipeline([
            OneHotVectorizer(columns=['education']),
            FastLinearRegressor(feature=['induced', 'education'], label='age',
                                number_of_threads=1, shuffle=False)])
        self.regressor_model = regressor_pipeline.fit(self.regression_data)
        self.regressor_pfi = self.regressor_model.permutation_feature_importance(self.regression_data)

        ticket_path = get_dataset('gen_tickettrain').as_filepath()
        self.ranking_data = FileDataStream.read_csv(ticket_path)
        ranker_pipeline = Pipeline([
            ToKey(columns=['group']),
            LightGbmRanker(feature=['Class', 'dep_day', 'duration'],
                           label='rank', group_id='group',
                           random_state=0, number_of_threads=1)])
        self.ranker_model = ranker_pipeline.fit(self.ranking_data)
        self.ranker_pfi = self.ranker_model.permutation_feature_importance(self.ranking_data)

    def test_binary_classifier(self):
        assert_almost_equal(self.binary_pfi['AreaUnderRocCurve'].sum(), -0.140824, 6)
        assert_almost_equal(self.binary_pfi['PositivePrecision'].sum(), -0.482143, 6)
        assert_almost_equal(self.binary_pfi['PositiveRecall'].sum(), -0.0695652, 6)
        assert_almost_equal(self.binary_pfi['NegativePrecision'].sum(), -0.0139899, 6)
        assert_almost_equal(self.binary_pfi['NegativeRecall'].sum(), -0.00779221, 6)
        assert_almost_equal(self.binary_pfi['F1Score'].sum(), -0.126983, 6)
        assert_almost_equal(self.binary_pfi['AreaUnderPrecisionRecallCurve'].sum(), -0.19365, 5)

    def test_binary_classifier_from_loaded_model(self):
        model_path = get_temp_model_file()
        self.binary_model.save_model(model_path)
        loaded_model = Pipeline()
        loaded_model.load_model(model_path)
        pfi_from_loaded = loaded_model.permutation_feature_importance(self.classification_data)
        assert_frame_equal(self.binary_pfi, pfi_from_loaded)
        os.remove(model_path)

    def test_clasifier(self):
        assert_almost_equal(self.classifier_pfi['MacroAccuracy'].sum(), -0.0256352, 6)
        assert_almost_equal(self.classifier_pfi['LogLoss'].sum(), 0.158811, 6)
        assert_almost_equal(self.classifier_pfi['LogLossReduction'].sum(), -0.29449, 5)
        assert_almost_equal(self.classifier_pfi['PerClassLogLoss.0'].sum(), 0.0808459, 6)
        assert_almost_equal(self.classifier_pfi['PerClassLogLoss.1'].sum(), 0.419826, 6)

    def test_classifier_from_loaded_model(self):
        model_path = get_temp_model_file()
        self.classifier_model.save_model(model_path)
        loaded_model = Pipeline()
        loaded_model.load_model(model_path)
        pfi_from_loaded = loaded_model.permutation_feature_importance(self.classification_data)
        assert_frame_equal(self.classifier_pfi, pfi_from_loaded)
        os.remove(model_path)

    def test_regressor(self):
        assert_almost_equal(self.regressor_pfi['MeanAbsoluteError'].sum(), 0.504701, 6)
        assert_almost_equal(self.regressor_pfi['MeanSquaredError'].sum(), 5.59277, 5)
        assert_almost_equal(self.regressor_pfi['RootMeanSquaredError'].sum(), 0.553048, 6)
        assert_almost_equal(self.regressor_pfi['RSquared'].sum(), -0.203612, 6)

    def test_regressor_from_loaded_model(self):
        model_path = get_temp_model_file()
        self.regressor_model.save_model(model_path)
        loaded_model = Pipeline()
        loaded_model.load_model(model_path)
        pfi_from_loaded = loaded_model.permutation_feature_importance(self.regression_data)
        assert_frame_equal(self.regressor_pfi, pfi_from_loaded)
        os.remove(model_path)

    def test_ranker(self):
        assert_almost_equal(self.ranker_pfi['DCG@1'].sum(), -2.16404, 5)
        assert_almost_equal(self.ranker_pfi['DCG@2'].sum(), -3.5294, 4)
        assert_almost_equal(self.ranker_pfi['DCG@3'].sum(), -4.9721, 4)
        assert_almost_equal(self.ranker_pfi['NDCG@1'].sum(), -0.114286, 6)
        assert_almost_equal(self.ranker_pfi['NDCG@2'].sum(), -0.198631, 6)
        assert_almost_equal(self.ranker_pfi['NDCG@3'].sum(), -0.236544, 6)

    def test_ranker_from_loaded_model(self):
        model_path = get_temp_model_file()
        self.ranker_model.save_model(model_path)
        loaded_model = Pipeline()
        loaded_model.load_model(model_path)
        pfi_from_loaded = loaded_model.permutation_feature_importance(self.ranking_data)
        assert_frame_equal(self.ranker_pfi, pfi_from_loaded)
        os.remove(model_path)

if __name__ == '__main__':
    unittest.main()
