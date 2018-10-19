# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import unittest

import pandas
import pandas as pd
from nimbusml import Pipeline as nimbusmlPipeline
from nimbusml.feature_extraction.categorical import OneHotVectorizer
from nimbusml.linear_model import FastLinearBinaryClassifier
from nimbusml.preprocessing.schema import ColumnConcatenator
from nimbusml.preprocessing.text import CharTokenizer
from sklearn.pipeline import Pipeline


class TestErrors(unittest.TestCase):

    def test_error_wrong_column_name(self):
        train = pandas.DataFrame(dict(Label=[1.0, 0.0, 1.0],
                                      f1=['a', 'b', 'a'],
                                      f2=['aa', 'aa', 'bb']))
        label = train['Label'].values
        train = train.drop('Label', axis=1)

        pipe = Pipeline(
            steps=[
                ('cat',
                 OneHotVectorizer() << [
                     'f1',
                     'f2',
                     'nothere']),
                ('linear',
                 FastLinearBinaryClassifier())])
        try:
            pipe.fit(train, label)
            assert False
        except Exception as e:
            if "unidentifiable C++ exception" in str(e):
                raise Exception(
                    'boost.python did not replace the exception.\n{0}'.format(
                        e))
            assert "Check the log for error messages" in str(e)

    @unittest.skip("System.NullReferenceException")
    def test_char_tokenizer(self):

        customer_reviews = pd.DataFrame(data=dict(review=[
            "I really did not like the taste of it",
            "It was surprisingly quite good!",
            "I will never ever ever go to that place again!!",
            "The best ever!! It was amazingly good and super fast",
            "I wish I had gone earlier, it was that great",
            "somewhat dissapointing. I'd probably wont try again",
            "Never visit again... rascals!"]))

        tokenize = CharTokenizer(['review'])
        concat = ColumnConcatenator() >> 'features' << [['review']]
        pipeline = nimbusmlPipeline([concat, tokenize])
        y = pipeline.fit_transform(customer_reviews)
        assert y is not None


if __name__ == '__main__':
    unittest.main()
