# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import unittest

import numpy as np
from nimbusml import FileDataStream
from nimbusml.datasets import get_dataset
from nimbusml.feature_extraction.categorical import OneHotHashVectorizer


class TestOneHotHashVectorizer(unittest.TestCase):

    def test_numeric_columns(self):
        path = get_dataset('infert').as_filepath()
        data = FileDataStream.read_csv(path, sep=',',
                                       numeric_dtype=np.float32)

        xf = OneHotHashVectorizer(
            columns={
                'edu': 'education',
                'in': 'induced',
                'sp': 'spontaneous'},
            number_of_bits=2)
        xf.fit_transform(data)

        xf = OneHotHashVectorizer(
            columns=[
                'education',
                'induced',
                'spontaneous'],
            number_of_bits=2)
        xf.fit_transform(data)


if __name__ == '__main__':
    unittest.main()
