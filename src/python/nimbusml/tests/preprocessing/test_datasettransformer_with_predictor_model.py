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
from nimbusml.linear_model import LogisticRegressionBinaryClassifier, OnlineGradientDescentRegressor
from nimbusml.naive_bayes import NaiveBayesClassifier
from nimbusml.preprocessing import DatasetTransformer
from nimbusml.preprocessing.filter import RangeFilter
from nimbusml import FileDataStream 

seed = 0

train_data = {'c0': ['a', 'b', 'a', 'b'],
              'c1': [1, 2, 3, 4],
              'c2': [2, 3, 4, 5]}
train_df = pd.DataFrame(train_data).astype({'c1': np.float64,
                                            'c2': np.float64})

test_data = {'c0': ['a', 'b', 'b'],
             'c1': [1.5, 2.3, 3.7],
             'c2': [2.2, 4.9, 2.7]}
test_df = pd.DataFrame(test_data).astype({'c1': np.float64,
                                          'c2': np.float64})

train_df_updated = train_df.drop(['c0'], axis=1)
test_df_updated = test_df.drop(['c0'], axis=1)
rf_max = 4.5


class TestDatasetTransformerWithPredictorModel(unittest.TestCase):

    def temp(self):
        transform_pipeline_1 = Pipeline([RangeFilter(min=0.0, max=rf_max) << 'c2'])
        transform_pipeline_1.fit(train_df_updated)
        combined_pipeline_1 = Pipeline([
            DatasetTransformer(transform_model=transform_pipeline_1.model),
            OnlineGradientDescentRegressor(label='c2', feature=['c1'])
        ], random_state=seed)
        combined_pipeline_1.fit(train_df_updated)
        result_1 = combined_pipeline_1.transform(test_df_updated)
        print(result_1)

    def temp2(self):
        transform_pipeline_2 = Pipeline([
            RangeFilter(min=0.0, max=rf_max) << 'c2', 
            OnlineGradientDescentRegressor(label='c2', feature=['c1'])
        ], random_state=seed)
        transform_pipeline_2.fit(train_df_updated)
        combined_pipeline_2 = Pipeline([
            DatasetTransformer(transform_model=transform_pipeline_2.model)
        ])
        combined_pipeline_2.fit(train_df_updated)
        result_2 = combined_pipeline_2.transform(test_df_updated)
        print(result_2)

    def temp3(self):
        transform_pipeline1 = Pipeline([
            RangeFilter(min=0.0, max=rf_max) << 'c2'
        ], random_state=seed)
        transform_pipeline1.fit(train_df)

        transform_pipeline2 = Pipeline([
            OneHotVectorizer() << 'c0',
            OnlineGradientDescentRegressor(label='c2', feature=['c0'])
            ], random_state=seed)
        transform_pipeline2.fit(train_df)

        combined_pipeline = Pipeline([
            DatasetTransformer(transform_model=transform_pipeline1.model),
            DatasetTransformer(transform_model=transform_pipeline2.model)
        ], random_state=seed)
        combined_pipeline.fit(train_df)

        os.remove(transform_pipeline1.model)
        os.remove(transform_pipeline2.model)

        result_2 = combined_pipeline.transform(test_df)
        print(result_2)

    def test_dataset_transformer_by_itself(self):
        transform_pipeline_1 = Pipeline([RangeFilter(min=0.0, max=rf_max) << 'c2'])
        transform_pipeline_1.fit(train_df_updated)
        combined_pipeline_1 = Pipeline([
            DatasetTransformer(transform_model=transform_pipeline_1.model),
            OnlineGradientDescentRegressor(label='c2', feature=['c1'])
        ], random_state=seed)
        combined_pipeline_1.fit(train_df_updated)
        result_1 = combined_pipeline_1.predict(test_df_updated)

        transform_pipeline_2 = Pipeline([
            RangeFilter(min=0.0, max=rf_max) << 'c2', 
            OnlineGradientDescentRegressor(label='c2', feature=['c1'])
        ], random_state=seed)
        transform_pipeline_2.fit(train_df_updated)
        combined_pipeline_2 = Pipeline([
            DatasetTransformer(transform_model=transform_pipeline_2.model)
        ])
        combined_pipeline_2.fit(train_df_updated)
        result_2 = combined_pipeline_2.transform(test_df_updated)

        os.remove(transform_pipeline_1.model)
        os.remove(transform_pipeline_2.model)

        self.assertEquals(result_1.sum().sum(), result_2.sum().sum())

    def test_dataset_transformer_at_end_with_transformers_before(self):
        std_pipeline = Pipeline([
            RangeFilter(min=0.0, max=rf_max) << 'c2',
            OneHotVectorizer() << 'c0',
            OnlineGradientDescentRegressor(label='c2', feature=['c0'])
        ], random_state=seed)

        std_pipeline.fit(train_df)
        result_1 = std_pipeline.predict(test_df)

        # Create combined pipeline
        transform_pipeline1 = Pipeline([
            RangeFilter(min=0.0, max=rf_max) << 'c2'
        ], random_state=seed)
        transform_pipeline1.fit(train_df)

        transform_pipeline2 = Pipeline([
            OneHotVectorizer() << 'c0',
            OnlineGradientDescentRegressor(label='c2', feature=['c0'])
            ], random_state=seed)
        transform_pipeline2.fit(train_df)

        combined_pipeline = Pipeline([
            DatasetTransformer(transform_model=transform_pipeline1.model),
            DatasetTransformer(transform_model=transform_pipeline2.model)
        ], random_state=seed)
        combined_pipeline.fit(train_df)

        os.remove(transform_pipeline1.model)
        os.remove(transform_pipeline2.model)

        result_2 = combined_pipeline.transform(test_df)

        self.assertEquals(result_1.sum().sum(), result_2.sum().sum())

    def test_dataset_transformer_at_end_with_transformers_before_and_predictor_after(self):
        transform_pipeline1 = Pipeline([
            RangeFilter(min=0.0, max=rf_max) << 'c2'
        ], random_state=seed)
        transform_pipeline1.fit(train_df)

        transform_pipeline2 = Pipeline([
            OneHotVectorizer() << 'c0',
            OnlineGradientDescentRegressor(label='c2', feature=['c0']),
            ], random_state=seed)
        transform_pipeline2.fit(train_df)

        combined_pipeline = Pipeline([
            DatasetTransformer(transform_model=transform_pipeline1.model),
            DatasetTransformer(transform_model=transform_pipeline2.model),
            OnlineGradientDescentRegressor(label='c1', feature=['c2'])
        ], random_state=seed)
        combined_pipeline.fit(train_df)

        os.remove(transform_pipeline1.model)

        result = combined_pipeline.predict(test_df)

        print(result)

        self.assertTrue(round(result["Score"][0], 6) == 0.750616)
        self.assertTrue(round(result["Score"][1], 6) == 0.801374)
        
    def test_get_fit_info(self):
        transform_pipeline = Pipeline([
            RangeFilter(min=0.0, max=4.5) << 'c2',
            OnlineGradientDescentRegressor(label='c2', feature=['c1'])
            ], random_state=seed)
        transform_pipeline.fit(train_df)

        combined_pipeline = Pipeline([DatasetTransformer(transform_model=transform_pipeline.model)])
        combined_pipeline.fit(train_df)

        info = combined_pipeline.get_fit_info(train_df)

        self.assertTrue(info[0][1]['name'] == 'DatasetTransformer')

if __name__ == '__main__':
    unittest.main()
