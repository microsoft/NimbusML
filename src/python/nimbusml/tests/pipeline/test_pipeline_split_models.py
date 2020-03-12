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
from nimbusml.preprocessing.schema import ColumnConcatenator, PrefixColumnConcatenator

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


class TestPipelineSplitModels(unittest.TestCase):

    def test_notvectorized_output_predictor_model(self):
        """
        This test verifies that outputted predictor model from 
        combined (with featurizers) pipeline runs successfully
        on featurized data with no vectors.
        """
        df = train_df.drop(['c0'], axis=1)

        # Create and fit a RangeFilter transform using the training
        # data and use it to transform the training data.
        transform_pipeline = Pipeline([RangeFilter(min=0.0, max=4.5) << 'c2'], random_state=seed)
        transform_pipeline.fit(df)
        df1 = transform_pipeline.transform(df)

        # Create and fit a combined model and spit out predictor model
        combined_pipeline = Pipeline([RangeFilter(min=0.0, max=4.5) << 'c2',
                            OnlineGradientDescentRegressor(feature=['c1'], label='c2')],
                           random_state=seed)
        combined_pipeline.fit(df, output_predictor_model=True)
        result_1 = combined_pipeline.predict(df)

        # Load predictor pipeline and score featurized data
        predictor_pipeline = Pipeline()
        predictor_pipeline.load_model(combined_pipeline.predictor_model)
        result_2 = predictor_pipeline.predict(df1)
                     
        self.assertEqual(result_1.loc[0, 'Score'], result_2.loc[0, 'Score'])
        self.assertEqual(result_1.loc[1, 'Score'], result_2.loc[1, 'Score'])

    def test_vectorized_output_predictor_model(self):
        """
        This test shows that outputted predictor model from 
        combined (with featurizers) pipeline fails to run
        on featurized data with vectors.
        """

        # Create and fit a OneHotVectorizer transform using the
        # training data and use it to transform the training data.
        transform_pipeline = Pipeline([OneHotVectorizer() << 'c0'], random_state=seed)
        transform_pipeline.fit(train_df)
        df = transform_pipeline.transform(train_df)

        # Create and fit a combined model and spit out predictor model
        combined_pipeline = Pipeline([OneHotVectorizer() << 'c0',
                            OnlineGradientDescentRegressor(label='c2')],
                           random_state=seed)
        combined_pipeline.fit(train_df, output_predictor_model=True)
        result_1 = combined_pipeline.predict(train_df)

        # Load predictor pipeline and score featurized data
        predictor_pipeline = Pipeline()
        predictor_pipeline.load_model(combined_pipeline.predictor_model)

        try:
            # This does not work because the input schema doesnt 
            # match. Input schema looks for vector 'c0' with slots 'a,b'
            # but featurized data has only columns 'c0.a' and 'c0.b'
            predictor_pipeline.predict(df)

        except Exception as e:
            pass
        else:
            self.fail()

    def test_vectorized_with_concat_output_predictor_model(self):
        """
        This test shows how to prepend ColumnConcatenator transform
        to outputted predictor model from combined (with featurizers) pipeline
        so it successfully runs on featurized data with vectors.
        """
        # Create and fit a OneHotVectorizer transform using the
        # training data and use it to transform the training data.
        transform_pipeline = Pipeline([OneHotVectorizer() << 'c0'], random_state=seed)
        transform_pipeline.fit(train_df)
        df = transform_pipeline.transform(train_df)

        # Create, fit and score with combined model. 
        # Output predictor model separately.
        combined_pipeline = Pipeline([OneHotVectorizer() << 'c0',
                            OnlineGradientDescentRegressor(label='c2')],
                           random_state=seed)
        combined_pipeline.fit(train_df, output_predictor_model=True)
        result_1 = combined_pipeline.predict(train_df)

        # train ColumnConcatenator on featurized data
        concat_pipeline = Pipeline([ColumnConcatenator(columns={'c0': ['c0.a', 'c0.b']})])
        concat_pipeline.fit(df)

        # Load predictor pipeline
        predictor_pipeline = Pipeline()
        predictor_pipeline.load_model(combined_pipeline.predictor_model)

        # combine concat and predictor models and score 
        combined_predictor_pipeline = Pipeline.combine_models(concat_pipeline, 
                                                    predictor_pipeline)
        result_2 = combined_predictor_pipeline.predict(df)
                     
        self.assertEqual(result_1.loc[0, 'Score'], result_2.loc[0, 'Score'])
        self.assertEqual(result_1.loc[1, 'Score'], result_2.loc[1, 'Score'])

    def test_vectorized_with_prefixconcat_output_predictor_model(self):
        """
        This test shows how to prepend ColumnConcatenator transform
        to outputted predictor model from combined (with featurizers) pipeline
        so it successfully runs on featurized data with vectors.
        """
        # Create and fit a OneHotVectorizer transform using the
        # training data and use it to transform the training data.
        transform_pipeline = Pipeline([OneHotVectorizer() << 'c0'], random_state=seed)
        transform_pipeline.fit(train_df)
        df = transform_pipeline.transform(train_df)

        # Create, fit and score with combined model. 
        # Output predictor model separately.
        combined_pipeline = Pipeline([OneHotVectorizer() << 'c0',
                            OnlineGradientDescentRegressor(label='c2')],
                           random_state=seed)
        combined_pipeline.fit(train_df, output_predictor_model=True)
        result_1 = combined_pipeline.predict(train_df)

        # train ColumnConcatenator on featurized data
        concat_pipeline = Pipeline([PrefixColumnConcatenator(columns={'c0': 'c0.'})])
        concat_pipeline.fit(df)

        # Load predictor pipeline
        predictor_pipeline = Pipeline()
        predictor_pipeline.load_model(combined_pipeline.predictor_model)

        # combine concat and predictor models and score 
        combined_predictor_pipeline = Pipeline.combine_models(concat_pipeline, 
                                                    predictor_pipeline)
        result_2 = combined_predictor_pipeline.predict(df)
                     
        self.assertEqual(result_1.loc[0, 'Score'], result_2.loc[0, 'Score'])
        self.assertEqual(result_1.loc[1, 'Score'], result_2.loc[1, 'Score'])

if __name__ == '__main__':
    unittest.main()

