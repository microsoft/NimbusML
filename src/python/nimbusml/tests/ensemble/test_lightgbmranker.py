# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
import platform
import unittest

import numpy as np
import pandas as pd
from nimbusml import FileDataStream
from nimbusml import Pipeline, Role
from nimbusml.datasets import get_dataset
from nimbusml.ensemble import LightGbmRanker
from nimbusml.preprocessing import ToKey
from sklearn.utils.testing import assert_almost_equal


class TestLightGbmRanker(unittest.TestCase):

    def test_lightgbmranker_asfilestream(self):
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

    def test_lightgbmranker_asdataframe(self):
        # Data file
        file_path = get_dataset("gen_tickettrain").as_filepath()

        df = pd.read_csv(file_path, encoding='utf-8')
        df['group'] = df['group'].astype(np.uint32)

        e = Pipeline([ToKey(columns={'rank': 'rank', 'group': 'group'}),
                      LightGbmRanker() << {
                          Role.Feature: ['Class', 'dep_day', 'duration'],
                          Role.Label: 'rank', Role.GroupId: 'group'}])

        e.fit(df)

        metrics, _ = e.test(df)
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

    def test_lightgbmranker_asdataframe_groupid(self):
        # Data file
        file_path = get_dataset("gen_tickettrain").as_filepath()

        df = pd.read_csv(file_path, encoding='utf-8')
        df['group'] = df['group'].astype(np.uint32)

        e = Pipeline(
            [ToKey(columns={'rank': 'rank', 'group': 'group'}), LightGbmRanker(
                feature=['Class', 'dep_day', 'duration'], label='rank',
                group_id='group')])

        e.fit(df)

        metrics, _ = e.test(df)
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

    def test_lightgbmranker_asfilestream_evaltyperanking(self):
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
