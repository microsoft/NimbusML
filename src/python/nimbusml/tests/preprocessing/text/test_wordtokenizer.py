# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
import unittest

import pandas
from nimbusml import Pipeline
from nimbusml.preprocessing.text import WordTokenizer


class TestWordTokenizer(unittest.TestCase):

    def test_wordtokenizer(self):
        customer_reviews = pandas.DataFrame(data=dict(review=[
            "I really did not like the taste of it",
            "It was surprisingly quite good!"]))

        tokenize = WordTokenizer(char_array_term_separators=[" ", "n"]) << 'review'
        pipeline = Pipeline([tokenize])

        tokenize.fit(customer_reviews)
        y = tokenize.transform(customer_reviews)

        self.assertEqual(y.shape, (2, 9))

        self.assertEqual(y.loc[0, 'review.3'], 'ot')
        self.assertEqual(y.loc[1, 'review.3'], 'gly')
        self.assertEqual(y.loc[1, 'review.6'], None)


if __name__ == '__main__':
    unittest.main()
