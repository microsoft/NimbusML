# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import unittest

from nimbusml import FileDataStream
from nimbusml.datasets import get_dataset
from nimbusml.preprocessing.schema import PrefixColumnConcatenator


class TestPrefixColumnConcatenator(unittest.TestCase):

    def test_prefix_columns_concatenator(self):
        data = get_dataset('iris').as_df()
        xf = PrefixColumnConcatenator(columns={'Spl': 'Sepal_', 'Pet': 'Petal_' })
        features = xf.fit_transform(data)

        assert features.shape == (150, 11)
        assert set(features.columns) == {
            'Sepal_Length',
            'Sepal_Width',
            'Petal_Length',
            'Petal_Width',
            'Label',
            'Species',
            'Setosa',
            'Spl.Sepal_Length',
            'Spl.Sepal_Width',
            'Pet.Petal_Length',
            'Pet.Petal_Width'}


if __name__ == '__main__':
    unittest.main()
