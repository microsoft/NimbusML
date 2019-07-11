# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import unittest

import numpy as np
import os
import sys
from nimbusml import Pipeline, FileDataStream, BinaryDataStream, DprepDataStream
from nimbusml.datasets import get_dataset
from nimbusml.preprocessing.normalization import MinMaxScaler
from sklearn.utils.testing import assert_true, assert_array_equal

def is_nan(x):
    return (x is np.nan or x != x)

def assert_2d_array_equal(actual, desired):
    if len(actual) != len(desired):
        assert_true(False, "arrays are of different lengths.")

    for i in range(len(actual)):
        if len(actual[i]) != len(desired[i]):
            assert_true(False, "arrays are of different lengths.")
        for y in range(len(actual[i])):
            if is_nan(actual[i][y]) and is_nan(desired[i][y]):
                continue
            assert_true(actual[i][y] == desired[i][y])

@unittest.skipIf(os.name == "posix" or sys.version_info[:2] != (3, 7), "azureml-dataprep is not installed.")
class TestDprep(unittest.TestCase):

    def test_fit_transform(self):
        import azureml.dataprep as dprep

        path = get_dataset('infert').as_filepath()
        dflow = dprep.auto_read_file(path=path)
        dprep_data = DprepDataStream(dflow)
        file_data = FileDataStream.read_csv(path)

        xf = MinMaxScaler(columns={'in': 'induced', 'sp': 'spontaneous'})
        pipe = Pipeline([xf])
        transformed_data = pipe.fit_transform(file_data)
        transformed_data1 = pipe.fit_transform(dprep_data)

        assert_array_equal(
            transformed_data.columns,
            transformed_data1.columns)
        assert_2d_array_equal(
            transformed_data.values,
            transformed_data1.values)

if __name__ == '__main__':
    unittest.main()
