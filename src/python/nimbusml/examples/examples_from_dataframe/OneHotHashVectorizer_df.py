###############################################################################
# OneHotHashVectorizer
import pandas
from nimbusml.feature_extraction.categorical import OneHotHashVectorizer
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

# OneHotHashVectorizer transform: the entire string is treated as a category.
# if output column name is same as input column, original input column values
# are replaced. number_of_bits=6 will hash into 2^6 -1 dimensions

y = train_reviews['like']
X = train_reviews.loc[:, train_reviews.columns != 'like']

cat = OneHotHashVectorizer(number_of_bits=6) << 'review'
X = cat.fit_transform(X)

# view the transformed numerical values and column names
print(X)

mymodel = LogisticRegressionBinaryClassifier().fit(X, y)

X_test = cat.transform(test_reviews)

scores = mymodel.predict(cat.transform(test_reviews))

# view the scores
print(scores)
