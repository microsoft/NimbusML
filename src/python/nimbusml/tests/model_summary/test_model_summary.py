# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import unittest

from nimbusml import FileDataStream
from nimbusml import Pipeline
from nimbusml.cluster import KMeansPlusPlus
from nimbusml.datasets import get_dataset
from nimbusml.decomposition import FactorizationMachineBinaryClassifier
from nimbusml.decomposition import PcaAnomalyDetector
from nimbusml.ensemble import FastForestBinaryClassifier
from nimbusml.ensemble import FastForestRegressor
from nimbusml.ensemble import FastTreesBinaryClassifier
from nimbusml.ensemble import FastTreesRegressor
from nimbusml.ensemble import FastTreesTweedieRegressor
from nimbusml.ensemble import GamBinaryClassifier
from nimbusml.ensemble import GamRegressor
from nimbusml.ensemble import LightGbmBinaryClassifier
from nimbusml.ensemble import LightGbmClassifier
from nimbusml.ensemble import LightGbmRegressor
from nimbusml.feature_extraction.categorical import OneHotVectorizer
from nimbusml.linear_model import AveragedPerceptronBinaryClassifier
from nimbusml.linear_model import FastLinearBinaryClassifier
from nimbusml.linear_model import FastLinearClassifier
from nimbusml.linear_model import FastLinearRegressor
from nimbusml.linear_model import LogisticRegressionBinaryClassifier
from nimbusml.linear_model import LogisticRegressionClassifier
from nimbusml.linear_model import OnlineGradientDescentRegressor
from nimbusml.linear_model import OrdinaryLeastSquaresRegressor
from nimbusml.linear_model import PoissonRegressionRegressor
from nimbusml.linear_model import SgdBinaryClassifier
#from nimbusml.linear_model import SymSgdBinaryClassifier
from nimbusml.multiclass import OneVsRestClassifier
from nimbusml.naive_bayes import NaiveBayesClassifier
from sklearn.utils.testing import assert_raises

train_file = get_dataset("uciadult_train").as_filepath()
categorical_columns = [
    'workclass',
    'education',
    'marital-status',
    'occupation',
    'relationship',
    'ethnicity',
    'sex',
    'native-country-region']
file_schema = 'sep=, col=label:R4:0 col=Features:R4:9-14 col=workclass:TX:1 ' \
              'col=education:TX:2 col=marital-status:TX:3 ' \
              'col=occupation:TX:4 col=relationship:TX:5 col=ethnicity:TX:6 ' \
              'col=sex:TX:7 col=native-country-region:TX:8 header+'
label_column = 'label'
learners = [
    AveragedPerceptronBinaryClassifier(),
    FastLinearBinaryClassifier(),
    FastLinearClassifier(),
    FastLinearRegressor(),
    LogisticRegressionBinaryClassifier(),
    LogisticRegressionClassifier(),
    OnlineGradientDescentRegressor(),
    SgdBinaryClassifier(),
    # Error on linux
    # Unable to load shared library 'SymSgdNative' or one of its dependencies
    #SymSgdBinaryClassifier(),
    OrdinaryLeastSquaresRegressor(),
    PoissonRegressionRegressor(),
    OneVsRestClassifier(FastLinearBinaryClassifier()),
    LightGbmClassifier(),
    GamRegressor(),
    GamBinaryClassifier(),
    PcaAnomalyDetector(),
    FactorizationMachineBinaryClassifier(),
    KMeansPlusPlus(),
    NaiveBayesClassifier()

    # Skipping these tests since they are throwing the following error:
    #   *** System.NotSupportedException: 'Column has variable length
    #   vector: CategoricalSplitFeatures. Not supported in python.
    #   Drop column before sending to Python
    #FastForestBinaryClassifier(),
    #FastForestRegressor(),
    #FastTreesBinaryClassifier(),
    #FastTreesRegressor(),
    #FastTreesTweedieRegressor(),
    #LightGbmRegressor(),
    #LightGbmBinaryClassifier(),
]

learners_not_supported = [
    #PcaTransformer(), # REVIEW: crashes
]


class TestModelSummary(unittest.TestCase):

    def test_model_summary(self):
        for learner in learners:
            pipeline = Pipeline(
                [OneHotVectorizer() << categorical_columns, learner])
            train_stream = FileDataStream(train_file, schema=file_schema)
            pipeline.fit(train_stream, label_column)
            pipeline.summary()

    @unittest.skip("No unsupported learners")
    def test_model_summary_not_supported(self):
        for learner in learners_not_supported:
            pipeline = Pipeline(
                [OneHotVectorizer() << categorical_columns, learner])
            train_stream = FileDataStream(train_file, schema=file_schema)
            pipeline.fit(train_stream, label_column)
            assert_raises(TypeError, pipeline.summary)

    def test_summary_called_back_to_back_on_predictor(self):
        """
        When a predictor is fit without using a Pipeline,
        calling summary() more than once should not throw
        an exception.
        """
        ols = OrdinaryLeastSquaresRegressor()
        ols.fit([1,2,3,4], [2,4,6,7])
        ols.summary()
        ols.summary()

    def test_pipeline_summary_is_refreshed_after_refitting(self):
        predictor = OrdinaryLeastSquaresRegressor(normalize='No', l2_regularization=0)
        pipeline = Pipeline([predictor])

        pipeline.fit([0,1,2,3], [1,2,3,4])
        summary1 = pipeline.summary()

        pipeline.fit([0,1,2,3], [2,5,8,11])
        summary2 = pipeline.summary()

        self.assertFalse(summary1.equals(summary2))

    def test_predictor_summary_is_refreshed_after_refitting(self):
        predictor = OrdinaryLeastSquaresRegressor(normalize='No', l2_regularization=0)

        predictor.fit([0,1,2,3], [1,2,3,4])
        summary1 = predictor.summary()

        predictor.fit([0,1,2,3], [2,5,8,11])
        summary2 = predictor.summary()

        self.assertFalse(summary1.equals(summary2))


if __name__ == '__main__':
    unittest.main()
