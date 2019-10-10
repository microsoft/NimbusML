# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
import unittest

import numpy as np
import pandas as pd
from nimbusml import Pipeline, FileDataStream
from nimbusml.datasets import get_dataset
from nimbusml.feature_extraction.categorical import OneHotVectorizer
from nimbusml.linear_model import OnlineGradientDescentRegressor
from nimbusml.preprocessing.filter import RangeFilter

train_data = {'c0': ['a', 'b', 'a', 'b'],
              'c1': [1, 2, 3, 4],
              'c2': [2, 3, 4, 5]}
train_df = pd.DataFrame(train_data).astype({'c1': np.float64,
                                            'c2': np.float64})


class TestPipelineGetSchema(unittest.TestCase):

    def test_get_schema_returns_correct_value_for_single_valued_columns(self):
        df = train_df.drop(['c0'], axis=1)

        pipeline = Pipeline([RangeFilter(min=0.0, max=4.5) << 'c2'])
        pipeline.fit(df)
        df = pipeline.transform(df)

        schema = pipeline.get_output_columns()

        self.assertTrue('c1' in schema)
        self.assertTrue('c2' in schema)

        self.assertEqual(len(schema), 2)

    def test_get_schema_returns_correct_value_for_vector_valued_columns(self):
        pipeline = Pipeline([OneHotVectorizer() << 'c0'])
        pipeline.fit(train_df)

        schema = pipeline.get_output_columns()

        self.assertTrue('c0.a' in schema)
        self.assertTrue('c0.b' in schema)
        self.assertTrue('c1' in schema)
        self.assertTrue('c2' in schema)

        self.assertEqual(len(schema), 4)

    def test_get_schema_does_not_work_when_predictor_is_part_of_model(self):
        df = train_df.drop(['c0'], axis=1)

        pipeline = Pipeline([OnlineGradientDescentRegressor(label='c2')])
        pipeline.fit(df)

        try:
            schema = pipeline.get_output_columns()
        except Exception as e:
            pass
        else:
            self.fail()


if __name__ == '__main__':
    unittest.main()

