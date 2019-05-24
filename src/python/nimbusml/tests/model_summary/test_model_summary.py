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
# from nimbusml.linear_model import SymSgdBinaryClassifier
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
    FastForestBinaryClassifier(),
    FastForestRegressor(),
    FastTreesBinaryClassifier(),
    FastTreesRegressor(),
    FastTreesTweedieRegressor(),
    LightGbmRegressor(),
    LightGbmBinaryClassifier(),
    AveragedPerceptronBinaryClassifier(),
    FastLinearBinaryClassifier(),
    FastLinearClassifier(),
    FastLinearRegressor(),
    LogisticRegressionBinaryClassifier(),
    LogisticRegressionClassifier(),
    OnlineGradientDescentRegressor(),
    SgdBinaryClassifier(),
    # SymSgdBinaryClassifier(),
    OrdinaryLeastSquaresRegressor(),
    PoissonRegressionRegressor()
]

learners_not_supported = [
    NaiveBayesClassifier(),
    # fix in nimbusml, needs to implement ICanGetSummaryAsIDataView
    KMeansPlusPlus(),
    # fix in nimbusml, needs to implement ICanGetSummaryAsIDataView
    # fix in nimbusml, needs to implement ICanGetSummaryAsIDataView
    FactorizationMachineBinaryClassifier(),
    PcaAnomalyDetector(),
    # fix in nimbusml, needs to implement ICanGetSummaryAsIDataView
    # PcaTransformer(), # REVIEW: crashes
    GamBinaryClassifier(),
    # fix in nimbusml, needs to implement ICanGetSummaryAsIDataView
    GamRegressor(),  # fix in nimbusml, needs to implement ICanGetSummaryAsIDataView
    LightGbmClassifier(),
    # fix in nimbusml, needs to implement ICanGetSummaryAsIDataView
    # LightGbmRanker(), # REVIEW: crashes
    # fix in nimbusml, needs to implement ICanGetSummaryAsIDataView
    OneVsRestClassifier(FastLinearBinaryClassifier()),
]


class TestModelSummary(unittest.TestCase):

    def test_model_summary(self):
        for learner in learners:
            pipeline = Pipeline(
                [OneHotVectorizer() << categorical_columns, learner])
            train_stream = FileDataStream(train_file, schema=file_schema)
            pipeline.fit(train_stream, label_column)
            pipeline.summary()

    def test_model_summary_not_supported(self):
        for learner in learners_not_supported:
            pipeline = Pipeline(
                [OneHotVectorizer() << categorical_columns, learner])
            train_stream = FileDataStream(train_file, schema=file_schema)
            pipeline.fit(train_stream, label_column)
            assert_raises(TypeError, pipeline.summary)


if __name__ == '__main__':
    unittest.main()
