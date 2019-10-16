# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
import os
import unittest

import numpy as np
import pandas as pd
from nimbusml import Pipeline, FileDataStream
from nimbusml.datasets import get_dataset
from nimbusml.feature_extraction.categorical import OneHotVectorizer
from nimbusml.linear_model import LogisticRegressionBinaryClassifier
from nimbusml.preprocessing import DatasetTransformer
from nimbusml.preprocessing.schema import PrefixColumnConcatenator
from nimbusml.preprocessing.schema import ColumnDropper
from numpy.testing import assert_equal

class TestCsrInput(unittest.TestCase):

    def test_predict_proba_on_csr(self):
        path = get_dataset('infert').as_filepath()
        data = FileDataStream.read_csv(path)
        cols = list(data.head(1).columns.values) # ordered data column names.
 
        # train featurizer
        featurization_pipeline = Pipeline([OneHotVectorizer(columns={'education': 'education'})])
        featurization_pipeline.fit(data)
        # Note: the relative order of all columns is still the same as in raw data.
        #print(featurization_pipeline.get_output_columns())

        # need to remove extra columns before getting csr_matrix featurized data as it wont have column name information.
        csr_featurization_pipeline = Pipeline([DatasetTransformer(featurization_pipeline.model), ColumnDropper() << ['case', 'row_num']])
        sparse_featurized_data = csr_featurization_pipeline.fit_transform(data, as_csr=True)
        # Note: the relative order of all columns is still the same.
        #print(csr_featurization_pipeline.get_output_columns())

        # train learner
        # Note: order & number of feature columns for learner (parameter 'feature') should be the same as in csr_matrix above
        cols.remove('row_num')
        cols.remove('case')
        feature_cols = cols
        #print(feature_cols)
        #['education', 'age', 'parity', 'induced', 'spontaneous', 'stratum', 'pooled.stratum']
        training_pipeline = Pipeline([DatasetTransformer(featurization_pipeline.model), LogisticRegressionBinaryClassifier(feature=feature_cols, label='case')])
        training_pipeline.fit(data, output_predictor_model=True)
 
        # load just a learner model
        predictor_pipeline = Pipeline()
        predictor_pipeline.load_model(training_pipeline.predictor_model)
        # see the order of Feature.* columns that get passed to learner algo
        #print(predictor_pipeline.get_output_columns())

        # use just a learner model on csr_matrix featurized data
        predictions = predictor_pipeline.predict_proba(sparse_featurized_data)
        assert_equal(len(predictions), 248)
        assert_equal(len(predictions[0]), 2)

        # get feature contributions
        fcc = predictor_pipeline.get_feature_contributions(sparse_featurized_data)
        assert_equal(fcc.shape, (248,30))

if __name__ == '__main__':
    unittest.main()

