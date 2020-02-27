# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
 
import json
import unittest
import six

import numpy as np
import pandas as pd
from nimbusml import Pipeline, Role
from nimbusml.cluster import KMeansPlusPlus
from nimbusml.ensemble import FastTreesRegressor, FastForestRegressor
from nimbusml.linear_model import FastLinearClassifier


class TestVariableColumn(unittest.TestCase):

    def verify_regressor_nodes(self, graph, label_name, features, trainer_name):
        nodes = graph['nodes']

        self.assertEqual(nodes[0]["Name"], "Transforms.OptionalColumnCreator")
        self.assertEqual(nodes[0]["Inputs"]["Column"], [label_name])

        self.assertEqual(nodes[1]["Name"], "Transforms.LabelToFloatConverter")
        self.assertEqual(nodes[1]["Inputs"]["LabelColumn"], label_name)

        self.assertEqual(nodes[2]["Name"], "Transforms.FeatureCombiner")
        self.assertEqual(nodes[2]["Inputs"]["Features"], features)

        self.assertEqual(nodes[3]["Name"], trainer_name)
        self.assertEqual(nodes[3]["Inputs"]["FeatureColumnName"], "Features")
        self.assertEqual(nodes[3]["Inputs"]["LabelColumnName"], label_name)

    def verify_classifier_nodes(self, graph, label_name, features, trainer_name):
        nodes = graph['nodes']

        self.assertEqual(nodes[0]["Name"], "Transforms.OptionalColumnCreator")
        self.assertEqual(nodes[0]["Inputs"]["Column"], [label_name])

        self.assertEqual(nodes[1]["Name"], "Transforms.LabelColumnKeyBooleanConverter")
        self.assertEqual(nodes[1]["Inputs"]["LabelColumn"], label_name)

        self.assertEqual(nodes[2]["Name"], "Transforms.FeatureCombiner")
        self.assertEqual(nodes[2]["Inputs"]["Features"], features)

        self.assertEqual(nodes[3]["Name"], trainer_name)
        self.assertEqual(nodes[3]["Inputs"]["FeatureColumnName"], "Features")
        self.assertEqual(nodes[3]["Inputs"]["LabelColumnName"], label_name)

    def test_label_column_defaults_to_label_when_no_label_column_in_input_data(self):
        train_data = {'c1': [2, 3, 4, 5], 'c2': [3, 4, 5, 6],
                      'c3': [4, 5, 6, 7], 'c4': [0, 1, 2, 1]}
        train_df = pd.DataFrame(train_data)

        predictor = FastForestRegressor()
        pipeline = Pipeline([predictor])
        result = json.loads(pipeline.fit(train_df, dry_run=True))

        self.verify_regressor_nodes(result, "Label",
                                    ["c1", "c2", "c3", "c4"],
                                    "Trainers.FastForestRegressor")

    def test_label_column_defaults_to_label_when_label_column_in_input_data(self):
        train_data = {'c1': [2, 3, 4, 5], 'c2': [3, 4, 5, 6],
                      'c3': [4, 5, 6, 7], 'Label': [0, 1, 2, 1]}
        train_df = pd.DataFrame(train_data)

        predictor = FastTreesRegressor()
        pipeline = Pipeline([predictor])
        result = json.loads(pipeline.fit(train_df, dry_run=True))

        self.verify_regressor_nodes(result, "Label",
                                    ["c1", "c2", "c3"],
                                    "Trainers.FastTreeRegressor")

    def test_label_column_specified_as_argument_without_features(self):
        train_data = {'c1': [2, 3, 4, 5], 'c2': [3, 4, 5, 6],
                      'd1': [4, 5, 6, 7], 'c4': [0, 1, 2, 1]}
        train_df = pd.DataFrame(train_data)

        predictor = FastForestRegressor(label='d1')
        pipeline = Pipeline([predictor])
        result = json.loads(pipeline.fit(train_df, dry_run=True))

        self.verify_regressor_nodes(result, "d1",
                                    ["c1", "c2", "c4"],
                                    "Trainers.FastForestRegressor")

    def test_label_column_specified_as_argument_with_features(self):
        train_data = {'c1': [2, 3, 4, 5], 'd1': [3, 4, 5, 6],
                      'c3': [4, 5, 6, 7], 'c4': [0, 1, 2, 1]}
        train_df = pd.DataFrame(train_data)

        predictor = FastForestRegressor(label='d1', feature=['c1', 'c3', 'c4'])
        pipeline = Pipeline([predictor])
        result = json.loads(pipeline.fit(train_df, dry_run=True))

        self.verify_regressor_nodes(result, "d1",
                                    ["c1", "c3", "c4"],
                                    "Trainers.FastForestRegressor")

    def test_label_column_specified_as_role_without_features(self):
        train_data = {'c1': [2, 3, 4, 5], 'd1': [3, 4, 5, 6],
                      'c3': [4, 5, 6, 7], 'c4': [0, 1, 2, 1]}
        train_df = pd.DataFrame(train_data)

        predictor = FastForestRegressor() << {Role.Label: 'd1'}
        pipeline = Pipeline([predictor])
        result = json.loads(pipeline.fit(train_df, dry_run=True))

        self.verify_regressor_nodes(result, "d1",
                                    ["c1", "c3", "c4"],
                                    "Trainers.FastForestRegressor")

    def test_label_column_specified_as_role_with_features(self):
        train_data = {'c1': [2, 3, 4, 5], 'd1': [3, 4, 5, 6],
                      'c3': [4, 5, 6, 7], 'c4': [0, 1, 2, 1]}
        train_df = pd.DataFrame(train_data)

        predictor = FastForestRegressor() << {
            Role.Label: 'd1',
            Role.Feature: ['c1', 'c4']
        }
        pipeline = Pipeline([predictor])
        result = json.loads(pipeline.fit(train_df, dry_run=True))

        self.verify_regressor_nodes(result, "d1",
                                    ["c1", "c4"],
                                    "Trainers.FastForestRegressor")

    def test_default_label_for_classifier_without_label_column(self):
        train_data = {'c1': [2, 3, 4, 5], 'c2': [3, 4, 5, 6],
                      'c3': [4, 5, 6, 7], 'c4': [0, 1, 2, 1]}
        train_df = pd.DataFrame(train_data)

        predictor = FastLinearClassifier()
        pipeline = Pipeline([predictor])
        result = json.loads(pipeline.fit(train_df, dry_run=True))

        self.verify_classifier_nodes(result, "Label",
                                     ['c1', 'c2', 'c3', 'c4'],
                                     "Trainers.StochasticDualCoordinateAscentClassifier")

    def test_default_label_for_classifier_with_label_column(self):
        train_data = {'c1': [2, 3, 4, 5], 'c2': [3, 4, 5, 6],
                      'c3': [4, 5, 6, 7], 'Label': [0, 1, 2, 1]}
        train_df = pd.DataFrame(train_data)

        predictor = FastLinearClassifier()
        pipeline = Pipeline([predictor])
        result = json.loads(pipeline.fit(train_df, dry_run=True))

        self.verify_classifier_nodes(result, "Label",
                                     ['c1', 'c2', 'c3'],
                                     "Trainers.StochasticDualCoordinateAscentClassifier")

    def test_label_column_for_classifier_specified_as_argument(self):
        train_data = {'c1': [2, 3, 4, 5], 'c2': [3, 4, 5, 6],
                      'c3': [4, 5, 6, 7], 'd1': [0, 1, 2, 1]}
        train_df = pd.DataFrame(train_data)

        predictor = FastLinearClassifier(label='d1')
        pipeline = Pipeline([predictor])
        result = json.loads(pipeline.fit(train_df, dry_run=True))

        self.verify_classifier_nodes(result, "d1",
                                     ['c1', 'c2', 'c3'],
                                     "Trainers.StochasticDualCoordinateAscentClassifier")

    def test_label_column_for_classifier_specified_as_argument_with_features(self):
        train_data = {'c1': [2, 3, 4, 5], 'c2': [3, 4, 5, 6],
                      'c3': [4, 5, 6, 7], 'd1': [0, 1, 2, 1]}
        train_df = pd.DataFrame(train_data)

        predictor = FastLinearClassifier(label='d1', feature=['c1', 'c2'])
        pipeline = Pipeline([predictor])
        result = json.loads(pipeline.fit(train_df, dry_run=True))

        self.verify_classifier_nodes(result, "d1",
                                     ['c1', 'c2'],
                                     "Trainers.StochasticDualCoordinateAscentClassifier")

    def test_label_column_for_classifier_specified_as_role_without_features(self):
        train_data = {'d1': [2, 3, 4, 5], 'c2': [3, 4, 5, 6],
                      'c3': [4, 5, 6, 7], 'c4': [0, 1, 2, 1]}
        train_df = pd.DataFrame(train_data)

        predictor = FastLinearClassifier() << {Role.Label: 'd1'}
        pipeline = Pipeline([predictor])
        result = json.loads(pipeline.fit(train_df, dry_run=True))

        self.verify_classifier_nodes(result, "d1",
                                     ['c2', 'c3', 'c4'],
                                     "Trainers.StochasticDualCoordinateAscentClassifier")

    def test_label_column_for_classifier_specified_as_role_with_features(self):
        train_data = {'c1': [2, 3, 4, 5], 'c2': [3, 4, 5, 6],
                      'c3': [4, 5, 6, 7], 'd1': [0, 1, 2, 1]}
        train_df = pd.DataFrame(train_data)

        predictor = FastLinearClassifier() << {
            Role.Label: 'd1',
            Role.Feature: ['c1', 'c4']
        }
        pipeline = Pipeline([predictor])
        result = json.loads(pipeline.fit(train_df, dry_run=True))

        self.verify_classifier_nodes(result, "d1",
                                     ['c1', 'c4'],
                                     "Trainers.StochasticDualCoordinateAscentClassifier")

    def test_non_label_based_predictor_does_not_have_label_column_automatically_removed(self):
        train_data = {'c1': [2, 3, 4, 5], 'c2': [3, 4, 5, 6],
                      'c3': [4, 5, 6, 7], 'Label': [0, 1, 2, 1]}
        train_df = pd.DataFrame(train_data)

        predictor = KMeansPlusPlus(n_clusters=5)
        pipeline = Pipeline([predictor])
        result = json.loads(pipeline.fit(train_df, dry_run=True))
        nodes = result['nodes']

        self.assertEqual(nodes[0]["Name"], "Transforms.FeatureCombiner")
        if six.PY2:
            self.assertItemsEqual(nodes[0]["Inputs"]["Features"], ['c1', 'c2', 'c3', 'Label'])
        else:
            self.assertCountEqual(nodes[0]["Inputs"]["Features"], ['c1', 'c2', 'c3', 'Label'])
        self.assertEqual(nodes[1]["Name"], "Trainers.KMeansPlusPlusClusterer")
        self.assertEqual(nodes[1]["Inputs"]["FeatureColumnName"], "Features")


if __name__ == '__main__':
    unittest.main()
