# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
import platform
import unittest

import numpy as np
import pandas as pd
from nimbusml.datasets import get_dataset
from nimbusml.ensemble import LightGbmRanker
from sklearn.pipeline import Pipeline

try:
    # To deal with multiple numpy versions.
    from numpy.testing import assert_almost_equal
except ImportError:
    from numpy.testing.utils import assert_almost_equal


class TestUciAdultScikit(unittest.TestCase):

    def test_lightgbmranker_scikit_noy(self):
        np.random.seed(0)

        file_path = get_dataset("gen_tickettrain").as_filepath()

        # Pure-nimbusml paradigm
        df = pd.read_csv(file_path, encoding='utf-8')
        df['group'] = df['group'].astype(np.uint32)

        # construct a scikit pipeline
        pipe = Pipeline([
            # the group_id column must be of key type
            ('lgbm', LightGbmRanker(feature=[
                'Class', 'dep_day', 'duration'], label='rank',
                group_id='group'))
        ])

        # Train Scikit Pipeline
        pipe.fit(df)

        # Predict
        scores = pipe.predict(df)
        assert_almost_equal(
            scores.values,
            self.nimbusml_per_instance_scores_sampleinputextraction(),
            decimal=6)

    def test_lightgbmranker_scikit(self):
        np.random.seed(0)

        file_path = get_dataset("gen_tickettrain").as_filepath()

        # Pure-nimbusml paradigm
        df = pd.read_csv(file_path, encoding='utf-8')
        df['group'] = df['group'].astype(np.uint32)

        # construct a scikit pipeline
        pipe = Pipeline([
            # the group_id column must be of key type
            ('lgbm', LightGbmRanker(feature=[
                'Class', 'dep_day', 'duration'], group_id='group'))
        ])

        # Train Scikit Pipeline
        X = df.drop(['rank'], axis=1)
        y = df['rank']
        pipe.fit(X, y)
        # Predict
        scores = pipe.predict(X)

        assert_almost_equal(
            scores.values,
            self.nimbusml_per_instance_scores_sampleinputextraction(),
            decimal=6)

    @staticmethod
    def nimbusml_per_instance_scores_sampleinputextraction():
        return np.array(
            [-0.12412092834711075, -0.12412092834711075, -0.12412092834711075,
             -0.3760618567466736, -0.3760618567466736,
             -0.3760618567466736, -0.19335459172725677, -0.19335459172725677,
             -0.12412092834711075,
             -0.12412092834711075, -0.3760618567466736, -0.3760618567466736,
             -0.3760618567466736, -0.19335459172725677,
             -0.19335459172725677, -0.19335459172725677, -0.157523050904274,
             0.20553988218307495, -0.06467318534851074,
             -0.06467318534851074, -0.14679445326328278, -0.04640105366706848,
             -0.3166141211986542, 0.34381651878356934,
             0.13630621135234833, -0.13390685617923737, -0.157523050904274,
             -0.157523050904274, 0.20553988218307495,
             -0.14679445326328278, -0.14679445326328278, -0.04640105366706848,
             0.34381651878356934, 0.13630621135234833,
             -0.4317062199115753, 0.20553988218307495, 0.20553988218307495,
             -0.4209776222705841, -0.04640105366706848,
             -0.04640105366706848, 0.069633349776268, 0.13630621135234833,
             0.13630621135234833, -0.19335459172725677,
             -0.19335459172725677, -0.4317062199115753, -0.4317062199115753,
             0.20553988218307495, 0.20553988218307495,
             -0.12412092834711075, -0.4209776222705841, -0.04640105366706848,
             -0.04640105366706848, -0.3760618567466736,
             -0.3760618567466736, 0.069633349776268, 0.069633349776268,
             0.069633349776268, 0.13630621135234833,
             0.13630621135234833, -0.3290014863014221, -0.3290014863014221,
             -0.1702122539281845, -0.1702122539281845,
             -0.1702122539281845, 0.30316615104675293, 0.30316615104675293,
             0.30316615104675293, 0.30316615104675293,
             0.30316615104675293, 0.30316615104675293, -0.3290014863014221,
             -0.3290014863014221, -0.3290014863014221,
             -0.3290014863014221, -0.21741074323654175, -0.1702122539281845,
             -0.1702122539281845, -0.4693516790866852,
             -0.1702122539281845, -0.1702122539281845, 0.30316615104675293,
             0.30316615104675293, 0.30316615104675293,
             0.02125929296016693]
        )


if __name__ == '__main__':
    unittest.main()
