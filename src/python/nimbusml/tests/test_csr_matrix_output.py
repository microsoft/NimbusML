# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import unittest

import numpy as np
import pandas as pd
from nimbusml import Pipeline
from nimbusml.feature_extraction.categorical import OneHotVectorizer
from nimbusml.preprocessing.schema import ColumnConcatenator, ColumnDropper
from scipy.sparse import csr_matrix 


class TestCsrMatrixOutput(unittest.TestCase):

    def test_column_dropped_output_produces_expected_result(self):
        train_data = {'c1': [1, 0, 0, 4],
                      'c2': [2, 3, 0, 5],
                      'c3': [3, 4, 5, 6]}
        train_df = pd.DataFrame(train_data).astype(np.float32)

        xf = ColumnDropper(columns=['c3'])
        xf.fit(train_df)
        result = xf.transform(train_df, as_csr=True)

        self.assertEqual(result.nnz, 5)
        self.assertTrue(type(result) == csr_matrix)
        result = pd.DataFrame(result.todense())

        train_data = {0: [1, 0, 0, 4],
                      1: [2, 3, 0, 5]}
        expected_result = pd.DataFrame(train_data).astype(np.float32)

        self.assertTrue(result.equals(expected_result))

    def test_fit_transform_produces_expected_result(self):
        train_data = {'c1': [1, 0, 0, 4],
                      'c2': [2, 3, 0, 5],
                      'c3': [3, 4, 5, 6]}
        train_df = pd.DataFrame(train_data).astype(np.float32)

        xf = ColumnDropper(columns=['c3'])
        result = xf.fit_transform(train_df, as_csr=True)

        self.assertEqual(result.nnz, 5)
        self.assertTrue(type(result) == csr_matrix)
        result = pd.DataFrame(result.todense())

        train_data = {0: [1, 0, 0, 4],
                      1: [2, 3, 0, 5]}
        expected_result = pd.DataFrame(train_data).astype(np.float32)

        self.assertTrue(result.equals(expected_result))

    def test_vector_column_combined_with_single_value_columns(self):
        train_data = {'c1': [1, 0, 0, 4],
                      'c2': [2, 3, 0, 5],
                      'c3': [3, 4, 5, 6]}
        train_df = pd.DataFrame(train_data).astype(np.float32)

        xf = ColumnConcatenator(columns={'features': ['c1', 'c2', 'c3']})
        xf.fit(train_df)
        result = xf.transform(train_df, as_csr=True)

        self.assertEqual(result.nnz, 18)
        self.assertTrue(type(result) == csr_matrix)
        result = pd.DataFrame(result.todense())

        train_data = {0: [1, 0, 0, 4],
                      1: [2, 3, 0, 5],
                      2: [3, 4, 5, 6],
                      3: [1, 0, 0, 4],
                      4: [2, 3, 0, 5],
                      5: [3, 4, 5, 6]}
        expected_result = pd.DataFrame(train_data).astype(np.float32)
        self.assertTrue(result.equals(expected_result))

    def test_sparse_vector_column(self):
        train_data = {'c0': ['a', 'b', 'a', 'b'],
                      'c1': ['c', 'd', 'd', 'c']}
        train_df = pd.DataFrame(train_data)

        xf = OneHotVectorizer(columns={'c0':'c0', 'c1':'c1'})
        xf.fit(train_df)
        expected_result = xf.transform(train_df)
        self.assertTrue(type(expected_result) == pd.DataFrame)

        result = xf.transform(train_df, as_csr=True)
        self.assertEqual(result.nnz, 8)
        self.assertTrue(type(result) == csr_matrix)

        result = pd.DataFrame(result.todense(), columns=['c0.a', 'c0.b', 'c1.c', 'c1.d'])

        self.assertTrue(result.equals(expected_result))

    def test_sparse_vector_column_combined_with_single_value_columns(self):
        train_data = {'c0': [0, 1, 0, 3],
                      'c1': ['a', 'b', 'a', 'b']}
        train_df = pd.DataFrame(train_data).astype({'c0': np.float32})

        xf = OneHotVectorizer(columns={'c1':'c1'})
        xf.fit(train_df)
        expected_result = xf.transform(train_df)
        self.assertTrue(type(expected_result) == pd.DataFrame)

        result = xf.transform(train_df, as_csr=True)
        self.assertEqual(result.nnz, 6)
        self.assertTrue(type(result) == csr_matrix)

        result = pd.DataFrame(result.todense(), columns=['c0', 'c1.a', 'c1.b'])

        self.assertTrue(result.equals(expected_result))

    def test_types_convertable_to_r4_get_output_as_r4(self):
        train_data = {'c1': [1, 0, 0, 4],
                      'c2': [2, 3, 0, 5],
                      'c3': [3, 4, 5, 6],
                      'c4': [4, 5, 6, 7]}
        train_df = pd.DataFrame(train_data).astype({'c1': np.ubyte,
                                                    'c2': np.short,
                                                    'c3': np.float32})

        xf = ColumnDropper(columns=['c4'])
        xf.fit(train_df)
        result = xf.transform(train_df, as_csr=True)

        self.assertTrue(type(result) == csr_matrix)
        self.assertEqual(result.nnz, 9)
        result = pd.DataFrame(result.todense())

        train_data = {0: [1, 0, 0, 4],
                      1: [2, 3, 0, 5],
                      2: [3, 4, 5, 6]}
        expected_result = pd.DataFrame(train_data).astype(np.float32)

        self.assertTrue(result.equals(expected_result))

        self.assertEqual(result.dtypes[0], np.float32)
        self.assertEqual(result.dtypes[1], np.float32)
        self.assertEqual(result.dtypes[2], np.float32)

    def test_types_convertable_to_r8_get_output_as_r8(self):
        large_int64 = 372036854775807
        train_data = {'c1': [1, 0, 0, 4],
                      'c2': [2, 3, 0, 5],
                      'c3': [3, 0, 5, 0],
                      'c4': [0, 5, 6, 7],
                      'c5': [0, 5, 0, large_int64],
                      'c6': [5, 6, 7, 8]}
        train_df = pd.DataFrame(train_data).astype({'c1': np.ubyte,
                                                    'c2': np.short,
                                                    'c3': np.float32,
                                                    'c4': np.float64,
                                                    'c5': np.int64})

        xf = ColumnDropper(columns=['c6'])
        xf.fit(train_df)
        result = xf.transform(train_df, as_csr=True)

        self.assertTrue(type(result) == csr_matrix)
        self.assertEqual(result.nnz, 12)
        result = pd.DataFrame(result.todense())

        train_data = {0: [1, 0, 0, 4],
                      1: [2, 3, 0, 5],
                      2: [3, 0, 5, 0],
                      3: [0, 5, 6, 7],
                      4: [0, 5, 0, large_int64]}
        expected_result = pd.DataFrame(train_data).astype(np.float64)

        self.assertTrue(result.equals(expected_result))

        self.assertEqual(result.dtypes[0], np.float64)
        self.assertEqual(result.dtypes[1], np.float64)
        self.assertEqual(result.dtypes[2], np.float64)
        self.assertEqual(result.dtypes[3], np.float64)

        self.assertEqual(result.loc[3, 4], large_int64)


if __name__ == '__main__':
    unittest.main()
