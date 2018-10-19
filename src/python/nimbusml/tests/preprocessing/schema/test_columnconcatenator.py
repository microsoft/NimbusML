# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import unittest

from nimbusml import FileDataStream
from nimbusml.datasets import get_dataset
from nimbusml.preprocessing.schema import ColumnConcatenator


class TestColumnConcatenator(unittest.TestCase):

    def test_columns_concatenator(self):
        path = get_dataset('infert').as_filepath()
        file_schema = 'sep=, col=id:TX:0 col=education:TX:1 col=age:R4:2 ' \
                      'col=parity:R4:3 col=induced:R4:4 col=case:R4:5 ' \
                      'col=spontaneous:R4:6 header=+'
        data = FileDataStream(path, schema=file_schema)
        xf = ColumnConcatenator(
            columns={
                'features': [
                    'age',
                    'parity',
                    'induced']})
        features = xf.fit_transform(data)
        assert features.shape == (248, 10)
        # columns ordering changed between 0.22 and 0.23
        assert set(
            features.columns) == {
            'age',
            'case',
            'education',
            'features.age',
            'features.induced',
            'features.parity',
            'id',
            'induced',
            'parity',
            'spontaneous'}

    def test_columns_concatenator_multi(self):
        path = get_dataset('infert').as_filepath()

        file_schema = 'sep=, col=id:TX:0 col=education:TX:1 col=age:R4:2 ' \
                      'col=parity:R4:3 col=induced:R4:4 col=case:R4:5 ' \
                      'col=spontaneous:R4:6 header=+'
        data = FileDataStream(path, schema=file_schema)

        xf = ColumnConcatenator(
            columns={
                'features': [
                    'age', 'parity', 'induced'], 'features2': [
                    'age', 'parity']})

        features = xf.fit_transform(data)

        assert features.shape == (248, 12)
        # columns ordering changed between 0.22 and 0.23
        assert set(
            features.columns) == {
            'age',
            'case',
            'education',
            'features.age',
            'features.induced',
            'features.parity',
            'features2.age',
            'features2.parity',
            'id',
            'induced',
            'parity',
            'spontaneous'}


if __name__ == '__main__':
    unittest.main()
