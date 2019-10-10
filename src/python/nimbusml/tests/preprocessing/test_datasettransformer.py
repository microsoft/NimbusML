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


class TestDatasetTransformer(unittest.TestCase):

    def test_same_schema_with_dataframe_input(self):
        train_df_updated = train_df.drop(['c0'], axis=1)
        test_df_updated = test_df.drop(['c0'], axis=1)

        rf_max = 4.5

        # Create reference pipeline
        std_pipeline = Pipeline([
            RangeFilter(min=0.0, max=rf_max) << 'c2',
            OnlineGradientDescentRegressor(label='c2', feature=['c1'])
        ], random_state=seed)

        std_pipeline.fit(train_df_updated)
        result_1 = std_pipeline.predict(test_df_updated)

        # Create combined pipeline
        transform_pipeline = Pipeline([RangeFilter(min=0.0, max=rf_max) << 'c2'])
        transform_pipeline.fit(train_df_updated)

        combined_pipeline = Pipeline([
            DatasetTransformer(transform_model=transform_pipeline.model),
            OnlineGradientDescentRegressor(label='c2', feature=['c1'])
        ], random_state=seed)
        combined_pipeline.fit(train_df_updated)

        os.remove(transform_pipeline.model)

        result_2 = combined_pipeline.predict(test_df_updated)

        self.assertTrue(result_1.equals(result_2))


    def test_different_schema_with_dataframe_input(self):
        # Create reference pipeline
        std_pipeline = Pipeline([
            OneHotVectorizer() << 'c0',
            OnlineGradientDescentRegressor(label='c2', feature=['c0', 'c1'])
        ], random_state=seed)

        std_pipeline.fit(train_df)
        result_1 = std_pipeline.predict(test_df)

        # Create combined pipeline
        transform_pipeline = Pipeline([OneHotVectorizer() << 'c0'], random_state=seed)
        transform_pipeline.fit(train_df)

        combined_pipeline = Pipeline([
            DatasetTransformer(transform_model=transform_pipeline.model),
            OnlineGradientDescentRegressor(label='c2', feature=['c0', 'c1'])
        ], random_state=seed)
        combined_pipeline.fit(train_df)

        os.remove(transform_pipeline.model)

        result_2 = combined_pipeline.predict(test_df)

        self.assertTrue(result_1.equals(result_2))


    def test_different_schema_with_filedatastream_input(self):
        train_filename = "train-data.csv"
        train_df.to_csv(train_filename, index=False, header=True)
        train_data_stream = FileDataStream.read_csv(train_filename, sep=',', header=True)

        test_filename = "test-data.csv"
        test_df.to_csv(test_filename, index=False, header=True)
        test_data_stream = FileDataStream.read_csv(test_filename, sep=',', header=True)

        # Create reference pipeline
        std_pipeline = Pipeline([
            OneHotVectorizer() << 'c0',
            OnlineGradientDescentRegressor(label='c2', feature=['c0', 'c1'])
        ], random_state=seed)

        std_pipeline.fit(train_data_stream)
        result_1 = std_pipeline.predict(test_data_stream)

        # Create combined pipeline
        transform_pipeline = Pipeline([OneHotVectorizer() << 'c0'], random_state=seed)
        transform_pipeline.fit(train_data_stream)

        combined_pipeline = Pipeline([
            DatasetTransformer(transform_model=transform_pipeline.model),
            OnlineGradientDescentRegressor(label='c2', feature=['c0', 'c1'])
        ], random_state=seed)
        combined_pipeline.fit(train_data_stream)

        os.remove(transform_pipeline.model)

        result_2 = combined_pipeline.predict(test_data_stream)

        self.assertTrue(result_1.equals(result_2))

        os.remove(train_filename)
        os.remove(test_filename)


    def test_combining_two_dataset_transformers(self):
        rf_max = 4.5

        # Create reference pipeline
        std_pipeline = Pipeline([
            RangeFilter(min=0.0, max=rf_max) << 'c2',
            OneHotVectorizer() << 'c0',
            OnlineGradientDescentRegressor(label='c2', feature=['c0', 'c1'])
        ], random_state=seed)

        std_pipeline.fit(train_df)
        result_1 = std_pipeline.predict(test_df)

        # Create combined pipeline
        transform_pipeline1 = Pipeline([RangeFilter(min=0.0, max=rf_max) << 'c2'])
        transform_pipeline1.fit(train_df)

        transform_pipeline2 = Pipeline([OneHotVectorizer() << 'c0'], random_state=seed)
        transform_pipeline2.fit(train_df)

        combined_pipeline = Pipeline([
            DatasetTransformer(transform_model=transform_pipeline1.model),
            DatasetTransformer(transform_model=transform_pipeline2.model),
            OnlineGradientDescentRegressor(label='c2', feature=['c0', 'c1'])
        ], random_state=seed)
        combined_pipeline.fit(train_df)

        os.remove(transform_pipeline1.model)
        os.remove(transform_pipeline2.model)

        result_2 = combined_pipeline.predict(test_df)

        self.assertTrue(result_1.equals(result_2))


    def test_get_fit_info(self):
        transform_pipeline = Pipeline([RangeFilter(min=0.0, max=4.5) << 'c2'])
        transform_pipeline.fit(train_df)

        combined_pipeline = Pipeline([
            DatasetTransformer(transform_model=transform_pipeline.model),
            OnlineGradientDescentRegressor(label='c2', feature=['c1'])
        ], random_state=seed)
        combined_pipeline.fit(train_df)

        info = combined_pipeline.get_fit_info(train_df)

        self.assertTrue(info[0][1]['name'] == 'DatasetTransformer')


if __name__ == '__main__':
    unittest.main()

