# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
import os
import unittest

import numpy as np
import pandas as pd
from nimbusml import Pipeline
from nimbusml.datasets import get_dataset
from nimbusml.feature_extraction.categorical import OneHotVectorizer
from nimbusml.linear_model import LogisticRegressionBinaryClassifier, OnlineGradientDescentRegressor
from nimbusml.preprocessing.filter import RangeFilter

seed = 0

train_data = {'c0': ['a', 'b', 'a', 'b'],
              'c1': [1, 2, 3, 4],
              'c2': [2, 3, 4, 5]}
train_df = pd.DataFrame(train_data).astype({'c1': np.float64,
                                            'c2': np.float64})

test_data = {'c0': ['a', 'b'],
             'c1': [1.5, 2.3],
             'c2': [2.2, 2.9]}
test_df = pd.DataFrame(test_data).astype({'c1': np.float64,
                                          'c2': np.float64})


class TestPipelineCombining(unittest.TestCase):

    def test_two_pipelines_created_using_dataframes_can_not_be_combined_when_the_schemas_are_different(self):
        """
        This test verifies that two models created using DataFrames
        can not be combined if the output schema of the first is
        different then the input schema of the second.
        NOTE: This issue only happens with Pipelines created and fit
        using dataframes. Pipelines created and fit using IDV binary
        streams do not have this issue (see the tests below).
        """
        # Create and fit a OneHotVectorizer transform using the
        # training data and use it to transform the training data.
        transform_pipeline = Pipeline([OneHotVectorizer() << 'c0'], random_state=seed)
        transform_pipeline.fit(train_df)
        df = transform_pipeline.transform(train_df)

        # Create and fit an OnlineGradientDescentRegressor using
        # the transformed training data from the previous step.
        predictor_pipeline = Pipeline([OnlineGradientDescentRegressor(label='c2')], random_state=seed)
        predictor_pipeline.fit(df)

        # Perform a prediction given the test data using
        # the transform and predictor defined previously.
        df = transform_pipeline.transform(test_df)
        result_1 = predictor_pipeline.predict(df)

        try:
            # This does not work because the output schema of the 
            combined_pipeline = Pipeline.combine_models(transform_pipeline,
                                                        predictor_pipeline)
        except Exception as e:
            pass
        else:
            self.fail()


    def test_two_pipelines_created_using_dataframes_can_be_combined_when_the_schemas_are_the_same(self):
        """
        This test verifies that two models created using DataFrames
        can be combined if the output schema of the first is the same
        as the input schema of the second.
        """
        df = train_df.drop(['c0'], axis=1)

        # Create and fit a RangeFilter transform using the training
        # data and use it to transform the training data.
        transform_pipeline = Pipeline([RangeFilter(min=0.0, max=4.5) << 'c2'], random_state=seed)
        transform_pipeline.fit(df)
        df = transform_pipeline.transform(df)

        # Create and fit an OnlineGradientDescentRegressor using
        # the transformed training data from the previous step.
        predictor_pipeline = Pipeline([OnlineGradientDescentRegressor(label='c2')],
                                      random_state=seed)
        predictor_pipeline.fit(df)

        # Perform a prediction given the test data using
        # the transform and predictor defined previously.
        df = transform_pipeline.transform(test_df)
        result_1 = predictor_pipeline.predict(df)

        df = test_df.drop(['c0'], axis=1)

        # Combine the above Pipelines in to one Pipeline and use
        # the new Pipeline to get predictions given the test data.
        combined_pipeline = Pipeline.combine_models(transform_pipeline,
                                                    predictor_pipeline)
        result_2 = combined_pipeline.predict(df)

        # Verify that the prediction from the combined Pipeline
        # matches the prediction from the original two Pipelines.
        self.assertEqual(result_1.loc[0, 'Score'], result_2.loc[0, 'Score'])
        self.assertEqual(result_1.loc[1, 'Score'], result_2.loc[1, 'Score'])


    def test_two_pipelines_created_using_idv_binary_data_can_be_combined_in_to_one_model(self):
        """
        This test verifies that two models can be combined
        even if the transform increases the number of columns.
        """
        # Create and fit a OneHotVectorizer transform using the
        # training data and use it to transform the training data.
        transform_pipeline = Pipeline([OneHotVectorizer() << 'c0'], random_state=seed)
        transform_pipeline.fit(train_df)
        df = transform_pipeline.transform(train_df, as_binary_data_stream=True)

        # Create and fit an OnlineGradientDescentRegressor using
        # the transformed training data from the previous step.
        predictor_pipeline = Pipeline([OnlineGradientDescentRegressor(label='c2', feature=['c0', 'c1'])],
                                      random_state=seed)
        predictor_pipeline.fit(df)

        # Perform a prediction given the test data using
        # the transform and predictor defined previously.
        df = transform_pipeline.transform(test_df, as_binary_data_stream=True)
        result_1 = predictor_pipeline.predict(df)

        # Combine the above Pipelines in to one Pipeline and use
        # the new Pipeline to get predictions given the test data.
        combined_pipeline = Pipeline.combine_models(transform_pipeline,
                                                    predictor_pipeline)
        result_2 = combined_pipeline.predict(test_df)

        # Verify that the prediction from the combined Pipeline
        # matches the prediction from the original two Pipelines.
        self.assertEqual(result_1.loc[0, 'Score'], result_2.loc[0, 'Score'])
        self.assertEqual(result_1.loc[1, 'Score'], result_2.loc[1, 'Score'])


    def test_three_pipelines_created_using_idv_binary_data_can_be_combined_in_to_one_model(self):
        """
        This test verifies that three models can be combined
        even if the transform increases the number of columns.
        """
        # Create and fit a RangeFilter transform using the training
        # data and use it to transform the training data.
        transform_pipeline_1 = Pipeline([RangeFilter(min=0.0, max=4.5) << 'c2'])
        df = transform_pipeline_1.fit_transform(train_df, as_binary_data_stream=True)

        # Create and fit a OneHotVectorizer transform using
        # the transformed data from the previous step and use it
        # to transform the data from the previous step.
        transform_pipeline_2 = Pipeline([OneHotVectorizer() << 'c0'], random_state=seed)
        transform_pipeline_2.fit(df)
        df = transform_pipeline_2.transform(df, as_binary_data_stream=True)

        # Create and fit an OnlineGradientDescentRegressor using
        # the transformed training data from the previous step.
        predictor_pipeline = Pipeline([OnlineGradientDescentRegressor(label='c2', feature=['c0', 'c1'])],
                                      random_state=seed)
        predictor_pipeline.fit(df)

        # Perform a prediction given the test data using
        # the transforms and predictor defined previously.
        df = transform_pipeline_1.transform(test_df, as_binary_data_stream=True)
        df = transform_pipeline_2.transform(df, as_binary_data_stream=True)
        result_1 = predictor_pipeline.predict(df)

        # Combine the above Pipelines in to one Pipeline and use
        # the new Pipeline to get predictions given the test data.
        combined_pipeline = Pipeline.combine_models(transform_pipeline_1,
                                                    transform_pipeline_2,
                                                    predictor_pipeline)
        result_2 = combined_pipeline.predict(test_df)

        # Verify that the prediction from the combined Pipeline
        # matches the prediction from the original two Pipelines.
        self.assertEqual(result_1.loc[0, 'Score'], result_2.loc[0, 'Score'])
        self.assertEqual(result_1.loc[1, 'Score'], result_2.loc[1, 'Score'])


    def test_combine_two_pipelines_created_from_model_files(self):
        """
        This test verifies that two models can be combined
        after they are loaded from disk in to new Pipelines.
        """
        # Create and fit a OneHotVectorizer transform using the
        # training data and use it to transform the training data.
        transform_pipeline_1 = Pipeline([OneHotVectorizer() << 'c0'], random_state=seed)
        transform_pipeline_1.fit(train_df)
        df = transform_pipeline_1.transform(train_df, as_binary_data_stream=True)

        # Create and fit an OnlineGradientDescentRegressor using
        # the transformed training data from the previous step.
        predictor_pipeline_1 = Pipeline([OnlineGradientDescentRegressor(label='c2', feature=['c0', 'c1'])],
                                        random_state=seed)
        predictor_pipeline_1.fit(df)

        # Perform a prediction given the test data using
        # the transform and predictor defined previously.
        df = transform_pipeline_1.transform(test_df, as_binary_data_stream=True)
        result_1 = predictor_pipeline_1.predict(df)

        # Use the model files stored in the Pipelines
        # to create new Pipelines (aka. create new Pipelines
        # using the model files stored on disk).
        transform_pipeline_2 = Pipeline()
        transform_pipeline_2.load_model(transform_pipeline_1.model)
        predictor_pipeline_2 = Pipeline()
        predictor_pipeline_2.load_model(predictor_pipeline_1.model)

        # Combine the newly created Pipelines in to one Pipeline
        # and use the it to get predictions given the test data.
        combined_pipeline = Pipeline.combine_models(transform_pipeline_2,
                                                    predictor_pipeline_2)
        result_2 = combined_pipeline.predict(test_df)

        # Verify that the prediction from the combined Pipeline
        # matches the prediction from the original two Pipelines.
        self.assertEqual(result_1.loc[0, 'Score'], result_2.loc[0, 'Score'])
        self.assertEqual(result_1.loc[1, 'Score'], result_2.loc[1, 'Score'])


    @unittest.skip("")
    def test_same_schema_at_join_point(self):
        pass


if __name__ == '__main__':
    unittest.main()

