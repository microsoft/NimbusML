# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
from __future__ import division

import unittest
import numpy as np
import pandas as pd

from nimbusml import DataSchema, FileDataStream, Pipeline, Role
from nimbusml.datasets import get_dataset
from nimbusml.ensemble import LightGbmRegressor, VotingRegressor
from nimbusml.feature_extraction.categorical import OneHotVectorizer
from nimbusml.linear_model import (LogisticRegressionClassifier,
                                   OrdinaryLeastSquaresRegressor,
                                   OnlineGradientDescentRegressor)
from nimbusml.model_selection import CV
from nimbusml.preprocessing.filter import RangeFilter
from nimbusml.preprocessing.missing_values import Indicator, Handler
from nimbusml.preprocessing.normalization import MeanVarianceScaler
from nimbusml.preprocessing.schema import ColumnDropper


train_data = {'c1': [2, 3, 4, 5],
              'c2': [20, 30.8, 39.2, 81]}
train_df = pd.DataFrame(train_data).astype({'c1': np.float32,
                                            'c2': np.float32})

test_data = {'c1': [2.5, 3.5],
             'c2': [1, 1]}
test_df = pd.DataFrame(test_data).astype({'c1': np.float32,
                                          'c2': np.float32})

olsrArgs = {
    'feature': ['c1'],
    'label': 'c2',
    'normalize': "Yes"
}

ogdArgs = {
    'feature': ['c1'],
    'label': 'c2',
    'shuffle': False,
    'number_of_iterations': 800,
    'learning_rate': 0.1,
    'normalize': "Yes"
}

lgbmArgs = {
    'feature':['c1'],
    'label':'c2',
    'random_state':1,
    'number_of_leaves':4,
    'minimum_example_count_per_leaf': 1,
    'normalize': 'Yes'
}


class TestVotingRegressor(unittest.TestCase):

    def test_ensemble_with_average_and_median_combiner(self):
        r1 = OrdinaryLeastSquaresRegressor(**olsrArgs)
        r1.fit(train_df)
        result1 = r1.predict(test_df)

        r2 = OnlineGradientDescentRegressor(**ogdArgs)
        r2.fit(train_df)
        result2 = r2.predict(test_df)

        r3 = LightGbmRegressor(**lgbmArgs)
        r3.fit(train_df)
        result3 = r3.predict(test_df)

        r1 = OrdinaryLeastSquaresRegressor(**olsrArgs)
        r2 = OnlineGradientDescentRegressor(**ogdArgs)
        r3 = LightGbmRegressor(**lgbmArgs)

        pipeline = Pipeline([VotingRegressor(estimators=[r1, r2, r3], combiner='Average')])
        pipeline.fit(train_df)
        result4 = pipeline.predict(test_df)

        average1 = (result1[0] + result2[0] + result3[0]) / 3
        average2 = (result1[1] + result2[1] + result3[1]) / 3
        self.assertAlmostEqual(average1, result4.loc[0, 'Score'], places=5)
        self.assertAlmostEqual(average2, result4.loc[1, 'Score'], places=5)

        r1 = OrdinaryLeastSquaresRegressor(**olsrArgs)
        r2 = OnlineGradientDescentRegressor(**ogdArgs)
        r3 = LightGbmRegressor(**lgbmArgs)

        pipeline = Pipeline([VotingRegressor(estimators=[r1, r2, r3], combiner='Median')])
        pipeline.fit(train_df)
        result4 = pipeline.predict(test_df)

        median1 = sorted([result1.loc[0], result2.loc[0], result3.loc[0]])[1]
        median2 = sorted([result1.loc[1], result2.loc[1], result3.loc[1]])[1]

        self.assertEqual(median1, result4.loc[0, 'Score'])
        self.assertEqual(median2, result4.loc[1, 'Score'])

    def test_ensemble_supports_user_defined_transforms(self):
        test2_df = test_df.copy(deep=True)
        test2_df = test2_df.append(pd.DataFrame({'c1': [9, 11], 'c2': [1, 1]}))

        r1 = OrdinaryLeastSquaresRegressor(**olsrArgs)
        r1.fit(train_df)
        result1 = r1.predict(test2_df)

        r2 = OnlineGradientDescentRegressor(**ogdArgs)
        r2.fit(train_df)
        result2 = r2.predict(test2_df)

        r3 = LightGbmRegressor(**lgbmArgs)
        r3.fit(train_df)
        result3 = r3.predict(test2_df)

        r1 = OrdinaryLeastSquaresRegressor(**olsrArgs)
        r2 = OnlineGradientDescentRegressor(**ogdArgs)
        r3 = LightGbmRegressor(**lgbmArgs)

        pipeline = Pipeline([
            RangeFilter(min=0, max=10, columns='c1'),
            VotingRegressor(estimators=[r1, r2, r3], combiner='Average')
        ])
        pipeline.fit(train_df)
        result4 = pipeline.predict(test2_df)

        self.assertEqual(len(result4), 3)

        average1 = (result1[0] + result2[0] + result3[0]) / 3
        average2 = (result1[1] + result2[1] + result3[1]) / 3
        average3 = (result1[2] + result2[2] + result3[2]) / 3
        self.assertAlmostEqual(average1, result4.loc[0, 'Score'], places=5)
        self.assertAlmostEqual(average2, result4.loc[1, 'Score'], places=5)
        self.assertAlmostEqual(average3, result4.loc[2, 'Score'], places=5)

    def test_ensemble_rejects_estimators_with_incorrect_type(self):
        r1 = OrdinaryLeastSquaresRegressor(**olsrArgs)
        r2 = OnlineGradientDescentRegressor(**ogdArgs)
        r3 = LogisticRegressionClassifier()
        try:
            vr = VotingRegressor(estimators=[r1, r2, r3], combiner='Average')
        except Exception as e:
            print(e)
        else:
            self.fail('VotingRegressor should only work with regressors.')

    @unittest.skip('Not implemented yet.')
    def test_ensemble_supports_output_predictor_model(self):
        test2_df = test_df.copy(deep=True)
        test2_df = test2_df.append(pd.DataFrame({'c1': [9, 11], 'c2': [1, 1]}),
                                   ignore_index=True)
        test2_df = test2_df.astype({'c1': np.float32, 'c2': np.float32})

        # Create a ground truth pipeline
        r1 = OrdinaryLeastSquaresRegressor(**olsrArgs)
        r2 = OnlineGradientDescentRegressor(**ogdArgs)
        combined_pipeline = Pipeline([RangeFilter(min=0.0, max=4.5) << 'c1',
                                      VotingRegressor(estimators=[r1, r2], combiner='Average')])
        combined_pipeline.fit(train_df)
        result_1 = combined_pipeline.predict(test2_df)

        # Create a duplicate pipeline but also request a predictor model
        r1 = OrdinaryLeastSquaresRegressor(**olsrArgs)
        r2 = OnlineGradientDescentRegressor(**ogdArgs)
        combined_pipeline = Pipeline([RangeFilter(min=0.0, max=4.5) << 'c1',
                                      VotingRegressor(estimators=[r1, r2], combiner='Average')])
        combined_pipeline.fit(train_df, output_predictor_model=True)
        result_2 = combined_pipeline.predict(test2_df)

        # Create a predictor model only pipeline
        predictor_pipeline = Pipeline()
        predictor_pipeline.load_model(combined_pipeline.predictor_model)
        result_3 = predictor_pipeline.predict(test2_df)

        # Verify the first rows are equal
        self.assertEqual(result_1.loc[0, 'Score'], result_2.loc[0, 'Score'])
        self.assertEqual(result_2.loc[0, 'Score'], result_3.loc[0, 'Score'])

        # Verify the second rows are equal
        self.assertEqual(result_1.loc[1, 'Score'], result_2.loc[1, 'Score'])
        self.assertEqual(result_2.loc[1, 'Score'], result_3.loc[1, 'Score'])

        # Verify the number of rows
        self.assertEqual(len(result_1), 2)
        self.assertEqual(len(result_2), 2)
        self.assertEqual(len(result_3), 4)

    def test_ensemble_supports_cv_with_user_defined_transforms(self):
        path = get_dataset("airquality").as_filepath()
        schema = DataSchema.read_schema(path)
        data = FileDataStream(path, schema)

        ind_args = {'Ozone_ind': 'Ozone', 'Solar_R_ind': 'Solar_R'}
        handler_args = {'Solar_R': 'Solar_R', 'Ozone': 'Ozone'}
        lgbm_args = {
            'feature': ['Ozone', 'Solar_R', 'Ozone_ind', 'Solar_R_ind', 'Temp'],
            'label': 'Wind',
            'normalize': 'Yes'
        }
        ols_args = {
            'feature': ['Ozone', 'Solar_R', 'Ozone_ind', 'Solar_R_ind', 'Temp'],
            'label': 'Wind',
            'normalize': 'Yes'
        }
        ogd_args = {
            'feature': ['Ozone', 'Solar_R', 'Ozone_ind', 'Solar_R_ind', 'Temp'],
            'label': 'Wind',
            'shuffle': False,
            'normalize': 'Yes'
        }

        for split_start in ['before_transforms', 'after_transforms']:
            pipeline_steps = [
                Indicator() << ind_args,
                Handler(replace_with='Mean') << handler_args,
                LightGbmRegressor(**lgbm_args)
            ]

            cv_results = CV(pipeline_steps).fit(data, split_start=split_start)
            l2_avg_lgbm = cv_results['metrics_summary'].loc['Average', 'L2(avg)']

            r1 = OrdinaryLeastSquaresRegressor(**ols_args)
            r2 = OnlineGradientDescentRegressor(**ogd_args)
            r3 = LightGbmRegressor(**lgbm_args)

            data = FileDataStream(path, schema)
            pipeline_steps = [
                Indicator() << ind_args,
                Handler(replace_with='Mean') << handler_args,
                VotingRegressor(estimators=[r1, r2, r3], combiner='Average')
            ]

            cv_results = CV(pipeline_steps).fit(data, split_start=split_start)
            l2_avg_ensemble = cv_results['metrics_summary'].loc['Average', 'L2(avg)']

            self.assertTrue(l2_avg_ensemble < l2_avg_lgbm)

    def test_ensemble_supports_cv_without_user_defined_transforms(self):
        path = get_dataset("airquality").as_filepath()
        schema = DataSchema.read_schema(path)
        data = FileDataStream(path, schema)

        ind_args = {'Ozone_ind': 'Ozone', 'Solar_R_ind': 'Solar_R'}
        handler_args = {'Solar_R': 'Solar_R', 'Ozone': 'Ozone'}
        lgbm_args = {
            'feature': ['Ozone', 'Solar_R', 'Ozone_ind', 'Solar_R_ind', 'Temp'],
            'label': 'Wind',
            'normalize': 'Yes'
        }
        ols_args = {
            'feature': ['Ozone', 'Solar_R', 'Ozone_ind', 'Solar_R_ind', 'Temp'],
            'label': 'Wind',
            'normalize': 'Yes'
        }
        ogd_args = {
            'feature': ['Ozone', 'Solar_R', 'Ozone_ind', 'Solar_R_ind', 'Temp'],
            'label': 'Wind',
            'shuffle': False,
            'normalize': 'Yes'
        }

        pipeline = Pipeline([
            Indicator() << ind_args,
            Handler(replace_with='Mean') << handler_args
        ])
        transformed_data = pipeline.fit_transform(data, as_binary_data_stream=True)

        pipeline_steps = [LightGbmRegressor(**lgbm_args)]
        cv_results = CV(pipeline_steps).fit(transformed_data)
        l2_avg_lgbm = cv_results['metrics_summary'].loc['Average', 'L2(avg)']

        r1 = OrdinaryLeastSquaresRegressor(**ols_args)
        r2 = OnlineGradientDescentRegressor(**ogd_args)
        r3 = LightGbmRegressor(**lgbm_args)

        pipeline_steps = [VotingRegressor(estimators=[r1, r2, r3], combiner='Average')]
        cv_results = CV(pipeline_steps).fit(transformed_data)
        l2_avg_ensemble = cv_results['metrics_summary'].loc['Average', 'L2(avg)']

        self.assertTrue(l2_avg_ensemble < l2_avg_lgbm)

    def test_ensemble_supports_get_fit_info(self):
        df = pd.DataFrame(dict(education=['A', 'B', 'A', 'B', 'A'],
                               workclass=['X', 'X', 'Y', 'Y', 'Y'],
                               yy=[1.1, 2.2, 1.24, 3.4, 3.4]))

        col_info = {'Feature': ['workclass', 'education'], Role.Label: 'new_y'}

        r1 = OrdinaryLeastSquaresRegressor(normalize="Yes") << col_info
        r2 = OnlineGradientDescentRegressor(normalize="Yes") << col_info
        r3 = LightGbmRegressor(normalize="Yes") << col_info

        pipeline = Pipeline([
            MeanVarianceScaler() << {'new_y': 'yy'},
            OneHotVectorizer() << ['workclass', 'education'],
            ColumnDropper() << 'yy',
            VotingRegressor(estimators=[r1, r2, r3], combiner='Average')
        ])

        info = pipeline.get_fit_info(df)

        last_info_node = info[0][-1]
        self.assertEqual(last_info_node['inputs'],
                         ['Feature:education,workclass', 'Label:new_y'])
        self.assertEqual(last_info_node['name'], 'VotingRegressor')
        self.assertTrue(isinstance(last_info_node['operator'], VotingRegressor))
        self.assertEqual(last_info_node['outputs'], ['Score'])
        self.assertEqual(last_info_node['schema_after'], ['Score'])
        self.assertEqual(last_info_node['type'], 'regressor')

    def test_data_role_info_has_been_removed_from_estimators(self):
        r1 = OrdinaryLeastSquaresRegressor(**olsrArgs)
        r2 = OnlineGradientDescentRegressor(**ogdArgs)
        r3 = LightGbmRegressor(**lgbmArgs)
        vr = VotingRegressor(estimators=[r1, r2, r3], combiner='Average')

        pipeline = Pipeline([vr])
        pipeline.fit(train_df)

        self.assertTrue(not hasattr(vr, 'feature_column_name'))

        self.assertTrue(not hasattr(vr.estimators[0], 'feature_column_name'))
        self.assertTrue(hasattr(vr.estimators[0], 'feature_column_name_'))

        self.assertTrue(not hasattr(vr.estimators[1], 'feature_column_name'))
        self.assertTrue(hasattr(vr.estimators[1], 'feature_column_name_'))

        self.assertTrue(not hasattr(vr.estimators[2], 'feature_column_name'))
        self.assertTrue(hasattr(vr.estimators[2], 'feature_column_name_'))


if __name__ == '__main__':
    unittest.main()
