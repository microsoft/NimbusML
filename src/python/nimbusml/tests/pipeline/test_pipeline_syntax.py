# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
import os
import unittest

import numpy as np
import pandas as pd
from nimbusml import Pipeline, FileDataStream
from nimbusml.datasets import get_dataset
from nimbusml.ensemble import LightGbmClassifier
from nimbusml.feature_extraction.text import NGramFeaturizer
from nimbusml.internal.entrypoints._ngramextractor_ngram import n_gram
from nimbusml.linear_model import LogisticRegressionBinaryClassifier

try:
    from pandas.testing import assert_frame_equal
except ImportError:
    # earlier versions
    from pandas.util.testing import assert_frame_equal


class TestPipelineSyntax(unittest.TestCase):

    def test_pipeline_name_error(self):
        trainData = pd.DataFrame(
            {
                "Sentiment": [
                    0,
                    1,
                    1,
                    0,
                    1,
                    1],
                "SentimentText": [
                    "this is train ",
                    "review ",
                    "sentence ",
                    "an apple",
                    "sentence 22",
                    "another one one one"]})
        NGramFeaturizer(
            word_feature_extractor=n_gram()).fit_transform(
            trainData[["SentimentText"]])

        msg = "Parameters ['NumLeaves', 'min_data', 'min_data_in_bin', " \
              "'minsplit'] are not allowed"
        with self.assertRaises(NameError, msg=msg):
            LightGbmClassifier(min_data=1, min_data_in_bin=1,
                               minimum_example_count_per_leaf=1,
                               minsplit=1, NumLeaves=2)
    @unittest.skip
    def test_pipeline_with_no_columns_raise(self):
        trainData = pd.DataFrame(
            {
                "Sentiment": [
                    0,
                    1,
                    1,
                    0,
                    1,
                    1],
                "SentimentText": [
                    "this is train ",
                    "review ",
                    "sentence ",
                    "an apple",
                    "sentence 22",
                    "another one one one"]})

        ppl = Pipeline([
            NGramFeaturizer(word_feature_extractor=n_gram()),
            LightGbmClassifier()
        ])
        assert ppl is not None

        # Bug 147697
        info = ppl.get_fit_info(
            trainData[["SentimentText"]], trainData["Sentiment"])
        assert len(info) == 2
        assert len(info[0]) == 3
        with self.assertRaises(RuntimeError):
            # Message
            # System.InvalidOperationException:
            # 'LightGBM Error, code is -1, error message is
            # 'Cannot construct Dataset since there are not useful features.
            # It should be at least two unique rows.
            # If the num_row (num_data) is small,
            # you can set min_data=1 and min_data_in_bin=1 to fix this.
            # Otherwise please make sure you are using the right dataset.'
            ppl.fit(trainData[["SentimentText"]], trainData["Sentiment"])

    def test_pipeline_with_no_columns(self):
        trainData = pd.DataFrame(
            {
                "Sentiment": [
                    0,
                    1,
                    1,
                    0,
                    1,
                    1],
                "SentimentText": [
                    "this is train ",
                    "review ",
                    "sentence ",
                    "an apple",
                    "sentence 22",
                    "another one one one"]})

        ppl = Pipeline([
            NGramFeaturizer(word_feature_extractor=n_gram()),
            LightGbmClassifier(minimum_example_count_per_leaf=1, minimum_example_count_per_group=1)
        ])
        assert ppl is not None

        # Bug 147697
        info = ppl.get_fit_info(
            trainData[["SentimentText"]], trainData["Sentiment"])
        assert len(info) == 2
        assert len(info[0]) == 3
        ppl.fit(trainData[["SentimentText"]], trainData["Sentiment"])

        ppl = Pipeline([
            NGramFeaturizer(word_feature_extractor=n_gram()),
            LightGbmClassifier(minimum_example_count_per_leaf=1, minimum_example_count_per_group=1)
        ])
        assert ppl is not None
        ppl.fit(trainData[["SentimentText"]], np.array(trainData["Sentiment"]))

    @unittest.skipIf(os.name != "nt", "not supported on this platform")
    def test_column_list_or_string(self):
        # Bug 142794
        data = pd.DataFrame({"Sentiment": [0,
                                           1,
                                           1,
                                           0,
                                           1,
                                           1],
                             "SentimentText": ["this is train ",
                                               "review ",
                                               "sentence ",
                                               "an apple",
                                               "sentence 22",
                                               "another one one one"]})
        data['SentimentText'] = data['SentimentText'].astype(str)
        featurizer = NGramFeaturizer(
            word_feature_extractor=n_gram()) << {
            "score": 'SentimentText'}
        featurizer = NGramFeaturizer(
            word_feature_extractor=n_gram()) << {
            "score": ['SentimentText']}
        featurizer = NGramFeaturizer(
            word_feature_extractor=n_gram(),
            columns=['SentimentText'])
        res1 = featurizer.fit_transform(data)
        featurizer = NGramFeaturizer(
            word_feature_extractor=n_gram()) << 'SentimentText'
        res2 = featurizer.fit_transform(data)
        assert_frame_equal(res1, res2)

    def test_with_or_without_pipeline(self):
        # Bug 227810
        # data input (as a FileDataStream)
        path = get_dataset('infert').as_filepath()

        file_schema = 'sep=, col=education:TX:1 col=Features:R4:2-4,6-8 ' \
                      'col=case:R4:5 header=+'
        data = FileDataStream(path, schema=file_schema)

        # without pipeline -- fails
        m = LogisticRegressionBinaryClassifier(
            feature=['Features'], label='case')
        m.fit(data)
        scores1 = m.predict(data)

        # with pipeline -- works
        m = Pipeline([LogisticRegressionBinaryClassifier(
            feature=['Features'], label='case')])
        m.fit(data)
        scores2 = m.predict(data)
        diff = np.abs(scores1.values.ravel() -
                      scores2[['PredictedLabel']].values.ravel())
        assert diff.sum() <= 2


if __name__ == '__main__':
    unittest.main()
