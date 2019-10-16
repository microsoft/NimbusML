# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
import os
import unittest
import pandas

from nimbusml import FileDataStream, Pipeline
from nimbusml.datasets import get_dataset
from nimbusml.feature_extraction.text import NGramExtractor
from nimbusml.preprocessing.text import CharTokenizer
from nimbusml.preprocessing.schema import ColumnDropper


class TestNGramExtractor(unittest.TestCase):

    def test_ngramfeaturizer(self):
        train_df = pandas.DataFrame(data=dict(review=['one', 'two']))

        pipeline = Pipeline([
            CharTokenizer(columns={'review_transform': 'review'}),
            NGramExtractor(ngram_length=3, all_lengths=False, columns={'ngrams': 'review_transform'}),
            ColumnDropper(columns=['review_transform', 'review'])
        ])

        result = pipeline.fit_transform(train_df)
        self.assertEqual(len(result.columns), 6)
        self.assertEqual(result.loc[0, 'ngrams.o|n|e'], 1.0)
        self.assertEqual(result.loc[1, 'ngrams.o|n|e'], 0.0)
        self.assertEqual(result.loc[0, 'ngrams.t|w|o'], 0.0)
        self.assertEqual(result.loc[1, 'ngrams.t|w|o'], 1.0)


if __name__ == '__main__':
    unittest.main()
