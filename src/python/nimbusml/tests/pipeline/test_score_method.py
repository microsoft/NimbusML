# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
import platform
import unittest

import numpy as np
from nimbusml import Pipeline, FileDataStream
from nimbusml.cluster import KMeansPlusPlus
from nimbusml.datasets import get_dataset
from nimbusml.ensemble import FastTreesRegressor, LightGbmRanker
from nimbusml.linear_model import LogisticRegressionBinaryClassifier
from nimbusml.linear_model import LogisticRegressionClassifier
from nimbusml.preprocessing import ToKey
from sklearn.model_selection import train_test_split
from sklearn.utils.testing import assert_almost_equal


class TestPiplineScoreMethod(unittest.TestCase):

    def test_score_binary(self):
        np.random.seed(0)
        df = get_dataset("iris").as_df()
        df.drop(['Species'], inplace=True, axis=1)
        df.Label = [1 if x == 1 else 0 for x in df.Label]
        X_train, X_test, y_train, y_test = \
            train_test_split(df.loc[:, df.columns != 'Label'], df['Label'])

        lr = LogisticRegressionBinaryClassifier(number_of_threads=1)
        e = Pipeline([lr])
        e.fit(X_train, y_train)
        metrics = e.score(X_test, y_test)
        print(metrics)
        assert_almost_equal(
            metrics,
            0.9801136363636364,
            decimal=5,
            err_msg="AUC should be %s" %
                    0.9801136363636364)

    def test_score_multiclass(self):
        np.random.seed(0)
        df = get_dataset("iris").as_df()
        df.drop(['Species'], inplace=True, axis=1)
        df.Label = [1 if x == 1 else 0 for x in df.Label]
        X_train, X_test, y_train, y_test = \
            train_test_split(df.loc[:, df.columns != 'Label'], df['Label'])

        lr = LogisticRegressionClassifier(number_of_threads=1)
        e = Pipeline([lr])
        e.fit(X_train, y_train.to_frame())
        metrics = e.score(X_test, y_test)
        print(metrics)
        assert_almost_equal(
            metrics,
            0.7631578947368421,
            decimal=5,
            err_msg="Accuracy(micro-avg) should be %s" %
                    0.7631578947368421)

    def test_score_regressor(self):
        np.random.seed(0)
        df = get_dataset("iris").as_df()
        df.drop(['Species'], inplace=True, axis=1)
        df.Label = [1 if x == 1 else 0 for x in df.Label]
        X_train, X_test, y_train, y_test = \
            train_test_split(df.loc[:, df.columns != 'Label'], df['Label'])

        lr = FastTreesRegressor(number_of_threads=1)
        e = Pipeline([lr])
        e.fit(X_train, y_train.to_frame())
        metrics = e.score(X_test, y_test)
        print(metrics)
        assert_almost_equal(
            metrics,
            0.814061733686017,
            decimal=5,
            err_msg="L1 loss should be %s" %
                    0.814061733686017)

    def test_score_clusterer(self):
        np.random.seed(0)
        df = get_dataset("iris").as_df()
        df.drop(['Species'], inplace=True, axis=1)
        df.Label = [1 if x == 1 else 0 for x in df.Label]
        X_train, X_test, y_train, y_test = \
            train_test_split(df.loc[:, df.columns != 'Label'], df['Label'])

        lr = KMeansPlusPlus(
            n_clusters=2,
            initialization_algorithm="Random",
            number_of_threads=1)
        e = Pipeline([lr])
        e.fit(X_train, y_train.to_frame())
        metrics = e.score(X_test, y_test)
        print(metrics)
        assert_almost_equal(
            metrics,
            0.36840763005544264,
            decimal=5,
            err_msg="NMI loss should be %s" %
                    0.36840763005544264)

    @unittest.skip("BUG: Not included in ML.NET yet")
    def test_score_anomalydetection(self):
        np.random.seed(0)
        df = get_dataset("iris").as_df().drop(['Label', 'Species'], axis=1)
        X_train, X_test = train_test_split(df)
        X_test.is_copy = False
        X_train = X_train[X_train['Setosa'] == 1]
        y_test = X_test['Setosa'].apply(lambda x: 1 if x == 0 else 0)
        X_train.drop(['Setosa'], axis=1, inplace=True)
        X_test.drop(['Setosa'], axis=1, inplace=True)
        svm = OneClassSvmAnomalyDetector() # noqa
        e = Pipeline([svm])
        e.fit(X_train)
        if e.nodes[-1].label_column_name_ is not None:
            raise ValueError("'{0}' should be None".format(
                e.nodes[-1].label_column_name_))
        assert y_test.name == 'Setosa'
        metrics = e.score(X_test, y_test)
        print(metrics)
        assert_almost_equal(
            metrics,
            1.0,
            decimal=5,
            err_msg="AUC should be %s" %
                    1.0)

    def test_score_ranking(self):
        # Data file
        file_path = get_dataset("gen_tickettrain").as_filepath()

        # Pure-nimbusml paradigm
        train_stream = FileDataStream.read_csv(file_path, encoding='utf-8')

        # pipeline
        pipeline = Pipeline([
            # the group_id column must be of key type
            ToKey(columns={'rank': 'rank', 'group': 'group'}),
            LightGbmRanker(
                feature=[
                    'Class',
                    'dep_day',
                    'duration'],
                label='rank',
                group_id='group')
        ])

        # train
        pipeline.fit(train_stream)

        # test
        eval_stream = FileDataStream.read_csv(file_path)
        metrics, _ = pipeline.test(eval_stream)
        assert_almost_equal(
            metrics['NDCG@1'][0],
            0.43571429,
            decimal=7,
            err_msg="NDCG@1 should be %s" %
                    0.43571429)
        assert_almost_equal(
            metrics['NDCG@2'][0],
            0.5128226,
            decimal=7,
            err_msg="NDCG@2 should be %s" %
                    0.5128226)
        assert_almost_equal(
            metrics['NDCG@3'][0],
            0.55168069,
            decimal=7,
            err_msg="NDCG@3 should be %s" %
                    0.55168069)
        assert_almost_equal(
            metrics['DCG@1'][0],
            4.688759,
            decimal=3,
            err_msg="DCG@1 should be %s" %
                    4.688759)
        assert_almost_equal(
            metrics['DCG@2'][0],
            9.012395,
            decimal=3,
            err_msg="DCG@2 should be %s" %
                    9.012395)
        assert_almost_equal(
            metrics['DCG@3'][0],
            11.446943,
            decimal=3,
            err_msg="DCG@3 should be %s" %
                    11.446943)


if __name__ == '__main__':
    unittest.main()
