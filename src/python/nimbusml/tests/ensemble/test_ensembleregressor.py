# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
import platform
import unittest

import numpy as np
from nimbusml.datasets import get_dataset
from nimbusml.ensemble import EnsembleRegressor
from nimbusml.ensemble.feature_selector import RandomFeatureSelector
from nimbusml.ensemble.output_combiner import RegressorMedian
from nimbusml.ensemble.subset_selector import RandomPartitionSelector
from nimbusml.ensemble.sub_model_selector import RegressorBestDiverseSelector
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.utils.testing import assert_greater, assert_less


class TestEnsembleRegressor(unittest.TestCase):

    def test_ensembleregressor(self):
        np.random.seed(0)

        df = get_dataset("airquality").as_df().fillna(0)
        df = df[df.Ozone.notnull()]

        X_train, X_test, y_train, y_test = train_test_split(
            df.loc[:, df.columns != 'Ozone'], df['Ozone'])

        # Train a model and score
        ensemble = EnsembleRegressor(num_models=3).fit(X_train, y_train)
        scores = ensemble.predict(X_test)

        r2 = r2_score(y_test, scores)
        assert_greater(r2, 0.12, "should be greater than %s" % 0.12)
        assert_less(r2, 0.13, "sum should be less than %s" % 0.13)

        ensemble_with_options = EnsembleRegressor(
            num_models=3,
            sampling_type = RandomPartitionSelector(
                feature_selector=RandomFeatureSelector(
                    features_selction_proportion=0.7)),
            sub_model_selector_type=RegressorBestDiverseSelector(),
            output_combiner=RegressorMedian()).fit(X_train, y_train)
        scores = ensemble_with_options.predict(X_test)

        r2 = r2_score(y_test, scores)
        assert_greater(r2, 0.0279, "R-Squared  should be greater than %s" % 0.0279)
        assert_less(r2, 0.03, "R-Squared should be less than %s" % 0.03)


if __name__ == '__main__':
    unittest.main()

