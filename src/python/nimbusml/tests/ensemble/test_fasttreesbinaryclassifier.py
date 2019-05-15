# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import unittest

from nimbusml import Pipeline, Role
from nimbusml.datasets import get_dataset
from nimbusml.ensemble import FastTreesBinaryClassifier
from nimbusml.preprocessing.schema import ColumnConcatenator
from numpy.testing import assert_array_almost_equal


class TestFastTreesBinaryClassifier(unittest.TestCase):

    def test_default_label(self):
        df = get_dataset("iris").as_df()
        df.drop(['Species'], inplace=True, axis=1)
        df.Label = [1 if x == 1 else 0 for x in df.Label]

        # 1
        pipeline = Pipeline(
            [
                ColumnConcatenator() << {
                    'Features': [
                        "Petal_Length",
                        "Sepal_Length"]},
                FastTreesBinaryClassifier(
                    number_of_trees=2) << {
                    Role.Label: 'Label',
                    Role.Feature: 'Features'}])

        model = pipeline.fit(df, verbose=0)
        probabilities0 = model.predict_proba(df)

        # 2
        pipeline = Pipeline([
            ColumnConcatenator() << {
                'Features': ["Petal_Length", "Sepal_Length"]},
            FastTreesBinaryClassifier(number_of_trees=2) << {
                Role.Feature: 'Features'}
        ])

        model = pipeline.fit(df, verbose=0)
        probabilities = model.predict_proba(df)
        assert_array_almost_equal(probabilities0, probabilities)

        # 3
        pipeline = Pipeline([
            ColumnConcatenator() << {
                'Features': ["Petal_Length", "Sepal_Length"]},
            FastTreesBinaryClassifier(number_of_trees=2)
        ])

        model = pipeline.fit(df, verbose=0)
        probabilities = model.predict_proba(df)
        assert_array_almost_equal(probabilities0, probabilities)

        # 4
        pipeline = Pipeline([
            ColumnConcatenator() << {
                'Features': ["Petal_Length", "Sepal_Length"]},
            FastTreesBinaryClassifier(number_of_trees=2) << {Role.Label: 'Label'}
        ])

        model = pipeline.fit(df, verbose=0)
        probabilities = model.predict_proba(df)
        assert_array_almost_equal(probabilities0, probabilities)


if __name__ == '__main__':
    unittest.main()
