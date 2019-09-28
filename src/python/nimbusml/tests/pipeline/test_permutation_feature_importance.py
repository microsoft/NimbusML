# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
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

adult_path = get_dataset('uciadult_train').as_filepath()
classification_data = FileDataStream.read_csv(adult_path)

infert_path = get_dataset('infert').as_filepath()
regression_data = FileDataStream.read_csv(infert_path)

ticket_path = get_dataset('gen_tickettrain').as_filepath()
ranking_data = FileDataStream.read_csv(ticket_path)

class TestPermutationFeatureImportance(unittest.TestCase):

    def test_binary_classifier(self):
        pipeline = Pipeline([
            OneHotVectorizer(columns=['education']),
            LogisticRegressionBinaryClassifier(
                feature=['age', 'education'], label='label')])
        model = pipeline.fit(classification_data)
        pfi = model.permutation_feature_importance(classification_data)
        assert_almost_equal(pfi['AreaUnderRocCurve'].sum(), -0.140824, 6)
        assert_almost_equal(pfi['PositivePrecision'].sum(), -0.482143, 6)
        assert_almost_equal(pfi['PositiveRecall'].sum(), -0.0695652, 6)
        assert_almost_equal(pfi['NegativePrecision'].sum(), -0.0139899, 6)
        assert_almost_equal(pfi['NegativeRecall'].sum(), -0.00779221, 6)
        assert_almost_equal(pfi['F1Score'].sum(), -0.126983, 6)
        assert_almost_equal(pfi['AreaUnderPrecisionRecallCurve'].sum(), -0.19374, 5)

    def test_clasifier(self):
        pipeline = Pipeline([
            OneHotVectorizer(columns=['education']),
            FastLinearClassifier(feature=['age', 'education'], label='label',
                                 number_of_threads=1, shuffle=False)])
        model = pipeline.fit(classification_data)
        pfi = model.permutation_feature_importance(classification_data)
        assert_almost_equal(pfi['MacroAccuracy'].sum(), -0.0256352, 6)
        assert_almost_equal(pfi['LogLoss'].sum(), 0.158811, 6)
        assert_almost_equal(pfi['LogLossReduction'].sum(), -0.29449, 5)
        assert_almost_equal(pfi['PerClassLogLoss.0'].sum(), 0.0808459, 6)
        assert_almost_equal(pfi['PerClassLogLoss.1'].sum(), 0.419826, 6)

    def test_regressor(self):
        pipeline = Pipeline([
            OneHotVectorizer(columns=['education']),
            FastLinearRegressor(feature=['induced', 'education'], label='age',
                                number_of_threads=1, shuffle=False)])
        model = pipeline.fit(regression_data)
        pfi = model.permutation_feature_importance(regression_data)
        assert_almost_equal(pfi['MeanAbsoluteError'].sum(), 0.504701, 6)
        assert_almost_equal(pfi['MeanSquaredError'].sum(), 5.59277, 5)
        assert_almost_equal(pfi['RootMeanSquaredError'].sum(), 0.553048, 6)
        assert_almost_equal(pfi['RSquared'].sum(), -0.203612, 6)

    def test_ranker(self):
        pipeline = Pipeline([
            ToKey(columns=['group']),
            LightGbmRanker(feature=['Class', 'dep_day', 'duration'],
                           label='rank', group_id='group')])
        model = pipeline.fit(ranking_data)
        pfi = model.permutation_feature_importance(ranking_data)
        assert_almost_equal(pfi['DCG@1'].sum(), -2.16404, 5)
        assert_almost_equal(pfi['DCG@2'].sum(), -3.5294, 4)
        assert_almost_equal(pfi['DCG@3'].sum(), -4.9721, 4)
        assert_almost_equal(pfi['NDCG@1'].sum(), -0.114286, 6)
        assert_almost_equal(pfi['NDCG@2'].sum(), -0.198631, 6)
        assert_almost_equal(pfi['NDCG@3'].sum(), -0.236544, 6)

if __name__ == '__main__':
    unittest.main()
