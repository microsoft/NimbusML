# -*- coding: utf-8 -*-
###############################################################################
# Example with NGramExtractor and LogisticRegressionBinaryClassifier
import pandas
from nimbusml import Pipeline
from nimbusml.feature_extraction.text import NGramExtractor
from nimbusml.linear_model import LogisticRegressionBinaryClassifier
from nimbusml.preprocessing.schema import ColumnConcatenator, ColumnDropper
from nimbusml.preprocessing.text import CharTokenizer

train_reviews = pandas.DataFrame(
    data=dict(
        review=[
            "This is great",
            "I hate it",
            "Love it",
            "Do not like it",
            "Really like it",
            "I hate it",
            "I like it a lot",
            "I kind of hate it",
            "I do like it",
            "I really hate it",
            "It is very good",
            "I hate it a bunch",
            "I love it a bunch",
            "I hate it",
            "I like it very much",
            "I hate it very much.",
            "I really do love it",
            "I really do hate it",
            "Love it!",
            "Hate it!",
            "I love it",
            "I hate it",
            "I love it",
            "I hate it",
            "I love it"],
        like=[
            True,
            False,
            True,
            False,
            True,
            False,
            True,
            False,
            True,
            False,
            True,
            False,
            True,
            False,
            True,
            False,
            True,
            False,
            True,
            False,
            True,
            False,
            True,
            False,
            True]))

test_reviews = pandas.DataFrame(
    data=dict(
        review=[
            "This is great",
            "I hate it",
            "Love it",
            "Really like it",
            "I hate it",
            "I like it a lot",
            "I love it",
            "I do like it",
            "I really hate it",
            "I love it"]))

y = train_reviews['like']
X = train_reviews.loc[:, train_reviews.columns != 'like']

pipeline = Pipeline([
    CharTokenizer(columns={'review_transform': 'review'}),
    NGramExtractor(ngram_length=3, all_lengths=False, columns={'ngrams': 'review_transform'}),
    ColumnDropper(columns=['review_transform', 'review'])
])
X = pipeline.fit_transform(X)

print(X.head())
#    ngrams.<␂>|T|h  ngrams.T|h|i  ngrams.h|i|s  ngrams.i|s|<␠>  ...  ngrams.i|t|!  ngrams.t|!|<␃>  ngrams.<␂>|H|a  ngrams.H|a|t
# 0             1.0           1.0           1.0             2.0  ...           0.0             0.0             0.0           0.0
# 1             0.0           0.0           0.0             0.0  ...           0.0             0.0             0.0           0.0
# 2             0.0           0.0           0.0             0.0  ...           0.0             0.0             0.0           0.0
# 3             0.0           0.0           0.0             0.0  ...           0.0             0.0             0.0           0.0
# 4             0.0           0.0           0.0             0.0  ...           0.0             0.0             0.0           0.0

model = LogisticRegressionBinaryClassifier().fit(X, y)

X_test = pipeline.transform(test_reviews)
result = model.predict(X_test)

print(result)
# 0     True
# 1    False
# 2     True
# 3     True
# 4    False
# 5     True
# 6     True
# 7     True
# 8    False
# 9     True
