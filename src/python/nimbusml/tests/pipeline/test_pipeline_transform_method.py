# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
import unittest

import pandas
from nimbusml import Pipeline, FileDataStream
from nimbusml.datasets import get_dataset
from nimbusml.feature_extraction.text import NGramFeaturizer

path = get_dataset("wiki_detox_train").as_filepath()
data = FileDataStream.read_csv(path, sep='\t')
df = data.to_df().head()
X = df['SentimentText']

class TestPipelineTransformMethod(unittest.TestCase):

    def test_transform_only_pipeline_transform_method(self):
        p = Pipeline([NGramFeaturizer(char_feature_extractor=None) << 'SentimentText'])
        p.fit(X)
        xf = p.transform(X)
        assert 'SentimentText.==rude==' in xf.columns

if __name__ == '__main__':
    unittest.main()
