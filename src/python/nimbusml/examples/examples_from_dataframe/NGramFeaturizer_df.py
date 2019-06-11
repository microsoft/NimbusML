###############################################################################
# Example with TextTransform and LogisticRegressionBinaryClassifier
import pandas
from nimbusml.feature_extraction.text import NGramFeaturizer
from nimbusml.internal.entrypoints._ngramextractor_ngram import n_gram
from nimbusml.linear_model import LogisticRegressionBinaryClassifier

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

ngram = NGramFeaturizer(word_feature_extractor=n_gram()) << 'review'
X = ngram.fit_transform(X)

# view the transformed numerical values and column names
# print(X.head())

mymodel = LogisticRegressionBinaryClassifier().fit(X, y)

X_test = ngram.transform(test_reviews)

scores = mymodel.predict(ngram.transform(test_reviews))

# view the scores
# print(scores.head())
