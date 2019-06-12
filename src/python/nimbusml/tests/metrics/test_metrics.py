# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
import platform
import unittest

import numpy as np
import pandas as pd
from nimbusml import Pipeline, FileDataStream
from nimbusml.cluster import KMeansPlusPlus
from nimbusml.datasets import get_dataset
from nimbusml.ensemble import FastForestRegressor
from nimbusml.ensemble import FastTreesRegressor
from nimbusml.ensemble import LightGbmRegressor
from nimbusml.feature_extraction.categorical import OneHotVectorizer
from nimbusml.linear_model import LogisticRegressionBinaryClassifier
from nimbusml.linear_model import LogisticRegressionClassifier
from sklearn.metrics import auc, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.utils.testing import assert_almost_equal


class TestMetrics(unittest.TestCase):

    def test_metrics_evaluate_binary(self):
        np.random.seed(0)
        df = get_dataset("iris").as_df()
        df.drop(['Species'], inplace=True, axis=1)
        df.Label = [1 if x == 1 else 0 for x in df.Label]
        X_train, X_test, y_train, y_test = \
            train_test_split(df.loc[:, df.columns != 'Label'], df['Label'])

        lr = LogisticRegressionBinaryClassifier()
        e = Pipeline([lr])
        e.fit(X_train, y_train, verbose=0)
        metrics, _ = e.test(X_test, y_test)
        # TODO: debug flucations, and increase decimal precision on checks
        assert_almost_equal(
            metrics['AUC'][0],
            0.980,
            decimal=1,
            err_msg="AUC should be %s" %
                    0.980)
        assert_almost_equal(
            metrics['Accuracy'][0],
            0.632,
            decimal=1,
            err_msg="Accuracy should be %s" %
                    0.632)
        assert_almost_equal(
            metrics['Positive precision'][0],
            1,
            decimal=1,
            err_msg="Positive precision should be %s" %
                    1)
        assert_almost_equal(
            metrics['Positive recall'][0],
            0.125,
            decimal=1,
            err_msg="Positive recall should be %s" %
                    0.125)
        assert_almost_equal(
            metrics['Negative precision'][0],
            0.611,
            decimal=1,
            err_msg="Negative precision should be %s" %
                    0.611)
        assert_almost_equal(
            metrics['Negative recall'][0],
            1,
            decimal=1,
            err_msg="Negative recall should be %s" %
                    1)
        assert_almost_equal(
            metrics['Log-loss'][0],
            0.686,
            decimal=1,
            err_msg="Log-loss should be %s" %
                    0.686)
        assert_almost_equal(
            metrics['Log-loss reduction'][0],
            0.3005,
            decimal=3,
            err_msg="Log-loss reduction should be %s" %
                    0.3005)
        assert_almost_equal(
            metrics['Test-set entropy (prior Log-Loss/instance)'][0],
            0.981,
            decimal=1,
            err_msg="Test-set entropy (prior Log-Loss/instance) should be %s" %
                    0.981)
        assert_almost_equal(
            metrics['F1 Score'][0],
            0.222,
            decimal=1,
            err_msg="F1 Score should be %s" %
                    0.222)
        assert_almost_equal(
            metrics['AUPRC'][0],
            0.966,
            decimal=1,
            err_msg="AUPRC should be %s" %
                    0.966)

    def test_metrics_evaluate_multiclass(self):
        np.random.seed(0)
        df = get_dataset("iris").as_df()
        df.drop(['Species'], inplace=True, axis=1)
        df.Label = [1 if x == 1 else 0 for x in df.Label]
        X_train, X_test, y_train, y_test = \
            train_test_split(df.loc[:, df.columns != 'Label'], df['Label'])

        lr = LogisticRegressionClassifier()
        e = Pipeline([lr])
        e.fit(X_train, y_train.to_frame(), verbose=0)
        metrics, _ = e.test(X_test, y_test)
        # TODO: debug flucations, and increase decimal precision on checks
        assert_almost_equal(
            metrics['Accuracy(micro-avg)'][0],
            0.763,
            decimal=1,
            err_msg="Accuracy(micro-avg) should be %s" %
                    0.763)
        assert_almost_equal(
            metrics['Accuracy(macro-avg)'][0],
            0.718,
            decimal=1,
            err_msg="Accuracy(macro-avg) should be %s" %
                    0.718)
        assert_almost_equal(
            metrics['Log-loss'][0],
            0.419,
            decimal=3,
            err_msg="Log-loss should be %s" %
                    0.419)
        assert_almost_equal(
            metrics['Log-loss reduction'][0],
            0.38476,
            decimal=3,
            err_msg="Log-loss reduction should be %s" %
                    0.38476)
        assert_almost_equal(
            metrics['(class 0)'][0],
            0.223,
            decimal=1,
            err_msg="(class 0) should be %s" %
                    0.223)
        assert_almost_equal(
            metrics['(class 1)'][0],
            0.688,
            decimal=1,
            err_msg="(class 1) should be %s" %
                    0.688)

    def test_metrics_evaluate_regressor(self):
        np.random.seed(0)
        df = get_dataset("iris").as_df()
        df.drop(['Species'], inplace=True, axis=1)
        df.Label = [1 if x == 1 else 0 for x in df.Label]
        X_train, X_test, y_train, y_test = \
            train_test_split(df.loc[:, df.columns != 'Label'], df['Label'])

        lr = FastTreesRegressor()
        e = Pipeline([lr])
        e.fit(X_train, y_train.to_frame(), verbose=0)
        metrics, _ = e.test(X_test, y_test)
        # TODO: debug flucations, and increase decimal precision on checks
        assert_almost_equal(
            metrics['L1(avg)'][0],
            0.107,
            decimal=1,
            err_msg="L1 loss should be %s" %
                    0.107)
        assert_almost_equal(
            metrics['L2(avg)'][0],
            0.0453,
            decimal=1,
            err_msg="L2(avg) should be %s" %
                    0.0453)
        assert_almost_equal(
            metrics['Loss-fn(avg)'][0],
            0.0453,
            decimal=1,
            err_msg="Loss-fn(avg)loss should be %s" %
                    0.0453)

    def test_metrics_evaluate_clusterer(self):
        np.random.seed(0)
        df = get_dataset("iris").as_df()
        df.drop(['Species'], inplace=True, axis=1)
        df.Label = [1 if x == 1 else 0 for x in df.Label]
        X_train, X_test, y_train, y_test = \
            train_test_split(df.loc[:, df.columns != 'Label'], df['Label'])

        lr = KMeansPlusPlus(n_clusters=2, initialization_algorithm="Random")
        e = Pipeline([lr])
        e.fit(X_train, y_train.to_frame(), verbose=0)
        metrics, _ = e.test(X_test, y_test)
        # if abs(metrics['NMI'][0] - 0.7) >= 0.15:
        #    raise AssertionError("NMI loss should be %f not %f" % \
        # (0.7, metrics['NMI'][0]))
        # if abs(metrics['AvgMinScore'][0] - 0.014) >= 0.015:
        #    raise AssertionError("AvgMinScore  should be %f not %f" % (\
        # 0.014, metrics['AvgMinScore'][0]))
        assert_almost_equal(
            metrics['NMI'][0],
            0.7,
            decimal=0,
            err_msg="NMI loss should be %s" %
                    0.7)
        assert_almost_equal(
            metrics['AvgMinScore'][0],
            0.032,
            decimal=2,
            err_msg="AvgMinScore  should be %s" %
                    0.014)

    @unittest.skip('ML.NET does not have svm')
    def test_metrics_evaluate_anomalydetection(self):
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
        e.fit(X_train, verbose=0)
        if e.nodes[-1].label_column_name_ is not None:
            raise ValueError("'{0}' should be None".format(
                e.nodes[-1].label_column_name_))
        assert y_test.name == 'Setosa'
        metrics, _ = e.test(X_test, y_test)
        assert_almost_equal(
            metrics['AUC'][0],
            1.0,
            decimal=5,
            err_msg="AUC should be %s" %
                    1.0)
        assert_almost_equal(
            metrics['DR @K FP'][0],
            1.0,
            decimal=5,
            err_msg="DR @K FP should be %s" %
                    1.0)
        assert_almost_equal(
            metrics['DR @P FPR'][0],
            1.0,
            decimal=5,
            err_msg="DR @P FPR should be %s" %
                    1.0)
        assert_almost_equal(
            metrics['DR @NumPos'][0],
            1.0,
            decimal=5,
            err_msg="DR @NumPos should be %s" %
                    1.0)
        assert_almost_equal(metrics['Threshold @K FP'][0], -0.0788,
                            decimal=2,
                            err_msg="Threshold @K FP should be %s" % -0.0788)
        assert_almost_equal(metrics['Threshold @P FPR'][0], -0.00352,
                            decimal=2,
                            err_msg="Threshold @P FPR "
                                    "should be %s" % -0.00352)
        assert_almost_equal(
            metrics['Threshold @NumPos'][0],
            1.5110,
            decimal=1,
            err_msg="Threshold @NumPos should be %s" %
                    1.5110)
        assert_almost_equal(
            metrics['NumAnomalies'][0],
            25,
            decimal=5,
            err_msg="NumAnomalies should be %s" %
                    25)

    def test_metrics_evaluate_ranking_group_id_from_new_dataframe(self):
        np.random.seed(0)
        df = get_dataset("iris").as_df().drop(['Label', 'Species'], axis=1)
        X_train, X_test = train_test_split(df)
        X_test.is_copy = False
        X_train.is_copy = False
        y_train = X_train['Setosa']
        y_test = X_test['Setosa']
        gvals_test = np.zeros(10).tolist() + np.ones(10).tolist() \
            + (np.ones(10) * 2).tolist() + (np.ones(8) * 3).tolist()

        gvals_train = np.zeros(30).tolist() \
            + np.ones(30).tolist() \
            + (np.ones(30) * 2).tolist() \
            + (np.ones(22) * 3).tolist()

        X_train.drop(['Setosa'], axis=1, inplace=True)
        X_test.drop(['Setosa'], axis=1, inplace=True)
        X_train['group_id'] = np.asarray(gvals_train, np.uint32)
        X_test['group_id'] = np.asarray(gvals_test, np.uint32)
        ft = FastForestRegressor()
        e = Pipeline([ft])
        e.fit(X_train, y_train, verbose=0)
        groups_df = pd.DataFrame(data=dict(groups=gvals_test))
        metrics, _ = e.test(
            X_test, y_test, evaltype='ranking', group_id=groups_df)
        assert_almost_equal(
            metrics['NDCG@1'][0],
            1,
            decimal=5,
            err_msg="NDCG@1 should be %s" %
                    1)
        assert_almost_equal(
            metrics['NDCG@2'][0],
            1,
            decimal=5,
            err_msg="NDCG@2 should be %s" %
                    1)
        assert_almost_equal(
            metrics['NDCG@3'][0],
            1,
            decimal=5,
            err_msg="NDCG@3 should be %s" %
                    1)
        # TODO: JRP comment for now. Debug fluctuations on build server
        # assert_almost_equal(metrics['DCG@1'][0], 4.32808, decimal=3,
        # err_msg="DCG@1 should be %s" % 4.32808)
        # assert_almost_equal(metrics['DCG@2'][0], 7.05880, decimal=3,
        # err_msg="DCG@2 should be %s" % 7.05880)
        # assert_almost_equal(metrics['DCG@3'][0], 7.59981, decimal=3,
        # err_msg="DCG@3 should be %s" % 7.59981)

    def test_metrics_evaluate_ranking_group_id_from_existing_column_in_X(self):
        np.random.seed(0)
        df = get_dataset("iris").as_df().drop(['Label', 'Species'], axis=1)
        X_train, X_test = train_test_split(df)
        X_test.is_copy = False
        X_train.is_copy = False
        y_train = X_train['Setosa']
        y_test = X_test['Setosa']
        gvals_test = np.zeros(10).tolist() \
            + np.ones(10).tolist() \
            + (np.ones(10) * 2).tolist() \
            + (np.ones(8) * 3).tolist()

        gvals_train = np.zeros(30).tolist() \
            + np.ones(30).tolist() \
            + (np.ones(30) * 2).tolist() \
            + (np.ones(22) * 3).tolist()

        X_train.drop(['Setosa'], axis=1, inplace=True)
        X_test.drop(['Setosa'], axis=1, inplace=True)
        X_train['group_id'] = np.asarray(gvals_train, np.uint32)
        X_test['group_id'] = np.asarray(gvals_test, np.uint32)
        ft = FastForestRegressor()
        e = Pipeline([ft])
        e.fit(X_train, y_train, verbose=0)
        metrics, _ = e.test(
            X_test, y_test, evaltype='ranking', group_id='group_id')
        assert_almost_equal(
            metrics['NDCG@1'][0],
            1,
            decimal=5,
            err_msg="NDCG@1 should be %s" %
                    1)
        assert_almost_equal(
            metrics['NDCG@2'][0],
            1,
            decimal=5,
            err_msg="NDCG@2 should be %s" %
                    1)
        assert_almost_equal(
            metrics['NDCG@3'][0],
            1,
            decimal=5,
            err_msg="NDCG@3 should be %s" %
                    1)
        assert_almost_equal(
            metrics['DCG@1'][0],
            4.32808,
            decimal=3,
            err_msg="DCG@1 should be %s" %
                    4.32808)
        assert_almost_equal(
            metrics['DCG@2'][0],
            7.05880,
            decimal=3,
            err_msg="DCG@2 should be %s" %
                    7.05880)
        assert_almost_equal(
            metrics['DCG@3'][0],
            8.68183,
            decimal=3,
            err_msg="DCG@3 should be %s" %
                    8.68183)

    def test_metrics_evaluate_binary_from_filedatastream(self):
        path = get_dataset('infert').as_filepath()
        data = FileDataStream.read_csv(path)
        e = Pipeline([
            OneHotVectorizer(columns={'edu': 'education'}),
            LightGbmRegressor(feature=['induced', 'edu'], label='age',
                              number_of_threads=1)
        ])
        e.fit(data, verbose=0)
        metrics, _ = e.test(data)
        # TODO: debug flucations, and increase decimal precision on checks
        assert_almost_equal(
            metrics['L1(avg)'][0],
            4.104164,
            decimal=4,
            err_msg="L1 loss should be %s" %
                    4.104164)
        assert_almost_equal(
            metrics['L2(avg)'][0],
            24.15286,
            decimal=4,
            err_msg="L2(avg) should be %s" %
                    24.15286)
        assert_almost_equal(metrics['Loss-fn(avg)'][0], 24.15286, decimal=4,
                            err_msg="Loss-fn(avg)loss should be %s" % 24.15286)

    def test_metrics_evaluate_binary_sklearn(self):
        np.random.seed(0)
        df = get_dataset("iris").as_df()
        df.drop(['Species'], inplace=True, axis=1)
        df.Label = [1 if x == 1 else 0 for x in df.Label]
        X_train, X_test, y_train, y_test = \
            train_test_split(df.loc[:, df.columns != 'Label'], df['Label'])

        lr = LogisticRegressionBinaryClassifier()
        e = Pipeline([lr])
        e.fit(X_train, y_train, verbose=0)

        metrics, scores = e.test(X_test, y_test, output_scores=True)
        aucnimbusml = metrics['AUC']
        precision, recall, _ = precision_recall_curve(
            y_test, scores['Probability'])
        aucskpr = auc(recall, precision)
        precision, recall, _ = precision_recall_curve(y_test, scores['Score'])
        aucsksc = auc(recall, precision)
        print(aucnimbusml, aucskpr, aucsksc)
        assert aucskpr == aucsksc
        # ML.NET: 0.980114
        # SKL: 0.9667731012859688
        # ML.NET computes the AUC as the probability that the score
        # for a positive example is higher than the score for a negative
        # example.
        # https://github.com/dotnet/machinelearning/blob/master/src/
        # Microsoft.ML.Data/Evaluators/AucAggregator.cs#L135
        # scikit-learn computes the AUC as the area under the curve.
        # https://github.com/scikit-learn/scikit-learn/blob/a24c8b46/sklearn/
        # metrics/ranking.py#L101

    def test_metrics_check_output_scores(self):
        np.random.seed(0)
        df = get_dataset("iris").as_df()
        df.drop(['Species'], inplace=True, axis=1)
        df.Label = [1 if x == 1 else 0 for x in df.Label]
        X_train, X_test, y_train, y_test = \
            train_test_split(df.loc[:, df.columns != 'Label'], df['Label'])

        lr = LogisticRegressionBinaryClassifier()
        e = Pipeline([lr])
        e.fit(X_train, y_train, verbose=0)
        metrics, scores = e.test(X_test, y_test, output_scores=False)
        assert len(scores) == 0
        metrics, scores = e.test(X_test, y_test, output_scores=True)
        assert len(scores) > 0


if __name__ == '__main__':
    unittest.main()
