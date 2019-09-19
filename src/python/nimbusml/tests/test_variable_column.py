# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
 
import unittest

import numpy as np
import pandas as pd
from nimbusml import Pipeline
from nimbusml.internal.entrypoints.transforms_variablecolumn import transforms_variablecolumn
from nimbusml.internal.utils.entrypoints import Graph, DataOutputFormat


class TestVariableColumn(unittest.TestCase):

    def to_variable_column(self, input, features=None, length_column_name=None):
        node = transforms_variablecolumn(data='$data',
                                         output_data='$output_data',
                                         features=features,
                                         length_column_name=length_column_name)

        graph_nodes = [node]
        graph = Graph(dict(data=''),
                      dict(output_data=''),
                      DataOutputFormat.DF,
                      *(graph_nodes))

        (out_model, out_data, out_metrics) = graph.run(verbose=True, X=input)
        return out_data

    def test_nonvariable_columns_are_returned_unchanged(self):
        train_data = {'c1': [2, 3, 4, 5],
                      'c2': [3, 4, 5, 6],
                      'c3': [4, 5, 6, 7],
                      'c4': [0, 1, 2, 1]}
        train_df = pd.DataFrame(train_data).astype({'c1': np.float64,
                                                    'c2': np.float64})

        result = self.to_variable_column(train_df, ['c1', 'c2'])

        self.assertTrue(result.loc[:, 'c3'].equals(train_df.loc[:, 'c3']))
        self.assertTrue(result.loc[:, 'c4'].equals(train_df.loc[:, 'c4']))

    def test_variable_columns_of_same_length_do_not_add_nans(self):
        train_data = {'c1': [2, 3, 4, 5],
                      'c2': [3, 4, 5, 6],
                      'c3': [4, 5, 6, 7]}
        train_df = pd.DataFrame(train_data).astype({'c1': np.float64,
                                                    'c2': np.float64})

        result = self.to_variable_column(train_df, ['c1', 'c2'])

        self.assertTrue(result.loc[:, 'c1.0'].equals(train_df.loc[:, 'c1']))
        self.assertTrue(result.loc[:, 'c1.1'].equals(train_df.loc[:, 'c2']))

    def test_variable_columns_with_different_lengths_return_nans(self):
        train_data = {'c1': [2, 3, 4, 5],
                      'c2': [3, 4, 5, 6],
                      'c3': [4, 5, 6, 7],
                      'c4': [0, 1, 2, 1]}
        train_df = pd.DataFrame(train_data).astype({'c1': np.float64,
                                                    'c2': np.float64})

        result = self.to_variable_column(train_df, ['c1', 'c2'], 'c4')

        expectedC1 = pd.Series([np.nan, 3, 4, 5]).astype(np.float64)
        expectedC2 = pd.Series([np.nan, np.nan, 5, np.nan]).astype(np.float64)

        self.assertTrue(result.loc[:, 'c1.0'].equals(expectedC1))
        self.assertTrue(result.loc[:, 'c1.1'].equals(expectedC2))

    def test_variable_columns_with_different_lengths_return_nans_when_no_other_columns_are_present(self):
        train_data = {'c1': [2, 3, 4, 5],
                      'c2': [3, 4, 5, 6],
                      'c3': [0, 1, 2, 1]}
        train_df = pd.DataFrame(train_data).astype({'c1': np.float64,
                                                    'c2': np.float64})

        result = self.to_variable_column(train_df, ['c1', 'c2'], 'c3')

        expectedC1 = pd.Series([np.nan, 3, 4, 5]).astype(np.float64)
        expectedC2 = pd.Series([np.nan, np.nan, 5, np.nan]).astype(np.float64)

        self.assertEqual(len(result.columns), 2)
        self.assertTrue(result.loc[:, 'c1.0'].equals(expectedC1))
        self.assertTrue(result.loc[:, 'c1.1'].equals(expectedC2))

    def test_variable_columns_are_converted_to_float32(self):
        """
        There are no integer nans so values that can be
        converted to float32 are converted to support nans.
        """
        types = [np.int8, np.int16, np.int32,
                 np.uint8, np.uint16, np.uint32,
                 np.float32]

        for type in types:
            train_data = {'c1': [2, 3, 4, 5], 'c2': [3, 4, 5, 6]}
            train_df = pd.DataFrame(train_data).astype(type);

            result = self.to_variable_column(train_df, ['c1', 'c2'])

            self.assertEqual(str(result.dtypes[0]), 'float32')
            self.assertEqual(str(result.dtypes[1]), 'float32')

    def test_variable_columns_are_converted_to_float64(self):
        """
        There are no integer nans so values that can be
        converted to float64 are converted to support nans.
        """
        types = [np.int64, np.uint64, np.float64]

        for type in types:
            train_data = {'c1': [2, 3, 4, 5], 'c2': [3, 4, 5, 6]}
            train_df = pd.DataFrame(train_data).astype(type);

            result = self.to_variable_column(train_df, ['c1', 'c2'])

            self.assertEqual(str(result.dtypes[0]), 'float64')
            self.assertEqual(str(result.dtypes[1]), 'float64')

    def test_column_with_all_vector_lengths_of_zero_returns_one_column_filled_with_nans(self):
        train_data = {'c1': [2, 3, 4, 5],
                      'c2': [3, 4, 5, 6],
                      'c3': [0, 0, 0, 0]}
        train_df = pd.DataFrame(train_data).astype({'c1': np.float64,
                                                    'c2': np.float64})

        result = self.to_variable_column(train_df, ['c1', 'c2'], 'c3')

        expectedC1 = pd.Series([np.nan, np.nan, np.nan, np.nan]).astype(np.float64)

        self.assertEqual(len(result.columns), 1)
        self.assertTrue(result.loc[:, 'c1.0'].equals(expectedC1))

    def test_variable_column_conversion_leaves_nans_untouched_if_they_already_exist_in_the_input(self):
        train_data = {'c1': [2, 3, np.nan, 5],
                      'c2': [3, np.nan, 5, 6],
                      'c3': [2, 2, 2, 1]}
        train_df = pd.DataFrame(train_data).astype({'c1': np.float64,
                                                    'c2': np.float64})

        result = self.to_variable_column(train_df, ['c1', 'c2'], 'c3')

        expectedC1 = pd.Series([2, 3, np.nan, 5]).astype(np.float64)
        expectedC2 = pd.Series([3, np.nan, 5, np.nan]).astype(np.float64)

        self.assertEqual(len(result.columns), 2)
        self.assertTrue(result.loc[:, 'c1.0'].equals(expectedC1))
        self.assertTrue(result.loc[:, 'c1.1'].equals(expectedC2))

if __name__ == '__main__':
    unittest.main()
