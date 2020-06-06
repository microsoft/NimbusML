# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import platform
import unittest

import numpy as np
import pandas as pd
import six
from nimbusml.datasets import get_dataset
from nimbusml.ensemble import EnsembleClassifier
from nimbusml.ensemble.feature_selector import RandomFeatureSelector
from nimbusml.ensemble.output_combiner import ClassifierVoting
from nimbusml.ensemble.subset_selector import RandomPartitionSelector
from nimbusml.ensemble.sub_model_selector import ClassifierBestDiverseSelector
from sklearn.model_selection import train_test_split
from sklearn.utils.testing import assert_greater


class TestEnsembleClassifier(unittest.TestCase):

    def test_ensembleclassifier(self):
        np.random.seed(0)
        df = get_dataset("iris").as_df()
        df.drop(['Species'], inplace=True, axis=1)

        X_train, X_test, y_train, y_test = \
            train_test_split(df.loc[:, df.columns != 'Label'], df['Label'])

        ensemble = EnsembleClassifier(num_models=3).fit(X_train, y_train, verbose=0)
        scores = ensemble.predict(X_test)
        scores = pd.to_numeric(scores)
        accuracy = np.mean(y_test == [i for i in scores])
        assert_greater(
            accuracy,
            0.947,
            "accuracy should be greater than %s" %
            0.948)

        ensemble_with_options = EnsembleClassifier(
            num_models=3,
            sampling_type = RandomPartitionSelector(
                feature_selector=RandomFeatureSelector(
                    features_selction_proportion=0.7)),
            sub_model_selector_type=ClassifierBestDiverseSelector(),
            output_combiner=ClassifierVoting()).fit(X_train, y_train)

        scores = ensemble.predict(X_test)
        scores = pd.to_numeric(scores)
        accuracy = np.mean(y_test == [i for i in scores])
        assert_greater(
            accuracy,
            0.578,
            "accuracy should be greater than %s" %
            0.579)


if __name__ == '__main__':
    unittest.main()

