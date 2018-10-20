# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import unittest

import pandas as pd
from nimbusml.feature_extraction.text import Sentiment

try:
    from pandas.testing import assert_frame_equal
except ImportError:
    # earlier versions
    from pandas.util.testing import assert_frame_equal


class TestSentiment(unittest.TestCase):

    @unittest.skip(
        "BUG: Error: *** System.InvalidOperationException: 'resourcePath', "
        "issue with ML.NET")
    def test_sentiment(self):
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
        featurizer = Sentiment() << {"score": 'SentimentText'}
        featurizer = Sentiment() << {"score": ['SentimentText']}
        featurizer = Sentiment(columns=['SentimentText'])
        res1 = featurizer.fit_transform(data)
        featurizer = Sentiment() << 'SentimentText'
        res2 = featurizer.fit_transform(data)
        assert_frame_equal(res1, res2)


if __name__ == '__main__':
    unittest.main()
