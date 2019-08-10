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
from nimbusml.preprocessing.filter import RangeFilter

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
        # and use it to get predictions given the test data.
        combined_pipeline = Pipeline.combine_models(transform_pipeline_2,
                                                    predictor_pipeline_2)
        result_2 = combined_pipeline.predict(test_df)

        # Verify that the prediction from the combined Pipeline
        # matches the prediction from the original two Pipelines.
        self.assertEqual(result_1.loc[0, 'Score'], result_2.loc[0, 'Score'])
        self.assertEqual(result_1.loc[1, 'Score'], result_2.loc[1, 'Score'])


    def test_passing_in_a_single_transform_returns_new_pipeline(self):
        transform = OneHotVectorizer() << 'c0'
        transform.fit(train_df)

        combined_pipeline = Pipeline.combine_models(transform,
                                                    contains_predictor=False)
        result = combined_pipeline.transform(test_df)

        self.assertEqual(len(result), 3)
        self.assertEqual(len(result.columns), 4)
        self.assertTrue(result.columns[0].startswith('c0.'))
        self.assertTrue(result.columns[1].startswith('c0.'))
        self.assertTrue(isinstance(combined_pipeline, Pipeline))


    def test_passing_in_a_single_predictor_returns_new_pipeline(self):
        train_dropped_df = train_df.drop(['c0'], axis=1)
        test_dropped_df = test_df.drop(['c0'], axis=1)

        predictor = OnlineGradientDescentRegressor(label='c2', feature=['c1'])
        predictor.fit(train_dropped_df)
        result_1 = predictor.predict(test_dropped_df)

        combined_pipeline = Pipeline.combine_models(predictor)
        result_2 = combined_pipeline.predict(test_dropped_df)

        self.assertEqual(result_1[0], result_2.loc[0, 'Score'])
        self.assertEqual(result_1[1], result_2.loc[1, 'Score'])
        self.assertTrue(isinstance(combined_pipeline, Pipeline))


    def test_passing_in_a_single_pipeline_returns_new_pipeline(self):
        pipeline = Pipeline([
            OneHotVectorizer() << 'c0',
            OnlineGradientDescentRegressor(label='c2', feature=['c0', 'c1'])
        ])
        pipeline.fit(train_df)
        result_1 = pipeline.predict(test_df)

        combined_pipeline = Pipeline.combine_models(pipeline)
        result_2 = combined_pipeline.predict(test_df)

        self.assertEqual(result_1.loc[0, 'Score'], result_2.loc[0, 'Score'])
        self.assertEqual(result_1.loc[1, 'Score'], result_2.loc[1, 'Score'])
        self.assertTrue(isinstance(combined_pipeline, Pipeline))


    def test_combine_transform_and_transform(self):
        transform_1 = RangeFilter(min=0.0, max=4.5) << 'c2'
        df = transform_1.fit_transform(train_df)

        transform_2 = OneHotVectorizer() << 'c0'
        transform_2.fit(df)

        df = transform_1.transform(test_df)
        result_1 = transform_2.transform(df)

        combined_pipeline = Pipeline.combine_models(transform_1,
                                                    transform_2,
                                                    contains_predictor=False)
        result_2 = combined_pipeline.transform(test_df)

        self.assertTrue(result_1.equals(result_2))


    def test_combine_transform_and_predictor(self):
        transform = OneHotVectorizer() << 'c0'
        df = transform.fit_transform(train_df, as_binary_data_stream=True)

        predictor = OnlineGradientDescentRegressor(label='c2', feature=['c0', 'c1'])
        predictor.fit(df)

        df = transform.transform(test_df, as_binary_data_stream=True)
        result_1 = predictor.predict(df)

        combined_pipeline = Pipeline.combine_models(transform, predictor)
        result_2 = combined_pipeline.predict(test_df)

        self.assertEqual(result_1[0], result_2.loc[0, 'Score'])
        self.assertEqual(result_1[1], result_2.loc[1, 'Score'])


    def test_combine_transform_and_pipeline(self):
        transform = RangeFilter(min=0.0, max=4.5) << 'c2'
        df = transform.fit_transform(train_df, as_binary_data_stream=True)

        pipeline = Pipeline([
            OneHotVectorizer() << 'c0',
            OnlineGradientDescentRegressor(label='c2', feature=['c0', 'c1'])
        ])
        pipeline.fit(df)

        df = transform.transform(test_df, as_binary_data_stream=True)
        result_1 = pipeline.predict(df)

        combined_pipeline = Pipeline.combine_models(transform, pipeline)
        result_2 = combined_pipeline.predict(test_df)

        self.assertTrue(result_1.equals(result_2))


    def test_combine_with_classifier_trained_with_y_arg(self):
        """
        Tests a sequence where the initial transform is computed
        using both X and y input args. Note, any steps after the
        initial transform will be operating on data where the X
        and y have been combined in to one dataset.
        """
        np.random.seed(0)

        df = get_dataset("infert").as_df()

        X = df.loc[:, df.columns != 'case']
        y = df['case']

        transform = OneHotVectorizer() << 'education_str'

        # Passing in both X and y
        df = transform.fit_transform(X, y, as_binary_data_stream=True)

        # NOTE: need to specify the label column here because the
        # feature and label data was joined in the last step.
        predictor = LogisticRegressionBinaryClassifier(label='case', feature=list(X.columns))
        predictor.fit(df)

        df = transform.transform(X, as_binary_data_stream=True)
        result_1 = predictor.predict(df)

        # Combine the models and perform a prediction
        combined_pipeline = Pipeline.combine_models(transform, predictor)
        result_2 = combined_pipeline.predict(X)

        result_2 = result_2['PredictedLabel'].astype(np.float64)
        self.assertTrue(result_1.equals(result_2))


    def test_combine_with_classifier_trained_with_joined_X_and_y(self):
        np.random.seed(0)

        infert_df = get_dataset("infert").as_df()
        feature_cols = [c for c in infert_df.columns if c != 'case']

        transform = OneHotVectorizer() << 'education_str'
        df = transform.fit_transform(infert_df, as_binary_data_stream=True)

        predictor = LogisticRegressionBinaryClassifier(label='case', feature=feature_cols)
        predictor.fit(df)

        df = transform.transform(infert_df, as_binary_data_stream=True)
        result_1 = predictor.predict(df)

        # Combine the models and perform a prediction
        combined_pipeline = Pipeline.combine_models(transform, predictor)
        result_2 = combined_pipeline.predict(infert_df)

        result_2 = result_2['PredictedLabel'].astype(np.float64)
        self.assertTrue(result_1.equals(result_2))


    def test_combine_with_classifier_trained_with_filedatastream(self):
        path = get_dataset('infert').as_filepath()

        data = FileDataStream.read_csv(path)

        transform = OneHotVectorizer(columns={'edu': 'education'})
        df = transform.fit_transform(data, as_binary_data_stream=True)

        feature_cols = ['parity', 'edu', 'age', 'induced', 'spontaneous', 'stratum', 'pooled.stratum']
        predictor = LogisticRegressionBinaryClassifier(feature=feature_cols, label='case')
        predictor.fit(df)

        data = FileDataStream.read_csv(path)
        df = transform.transform(data, as_binary_data_stream=True)
        result_1 = predictor.predict(df)

        data = FileDataStream.read_csv(path)
        combined_pipeline = Pipeline.combine_models(transform, predictor)
        result_2 = combined_pipeline.predict(data)

        result_1 = result_1.astype(np.int32)
        result_2 = result_2['PredictedLabel'].astype(np.int32)
        self.assertTrue(result_1.equals(result_2))


if __name__ == '__main__':
    unittest.main()

