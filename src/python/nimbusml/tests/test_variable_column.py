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

    def to_variable_column(self, input, features):
        node = transforms_variablecolumn(data='$data',
                                         output_data='$output_data',
                                         features=features)

        graph_nodes = [node]
        graph = Graph(dict(data=''),
                      dict(output_data=''),
                      DataOutputFormat.DF,
                      *(graph_nodes))

        (out_model, out_data, out_metrics) = graph.run(verbose=True, X=input)
        return out_data

    def test_column_dropped_output_produces_expected_result(self):
        train_data = {'c1': [1, 0, 0, 4],
                      'c2': [2, 3, 0, 5],
                      'c3': [3, 4, 5, 6]}
        train_df = pd.DataFrame(train_data).astype(np.float32)

        result = self.to_variable_column(train_df, ['c1', 'c2'])
        print(result)

if __name__ == '__main__':
    unittest.main()
