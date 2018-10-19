###############################################################################
# Example of MutualInformationSelector
import pandas
from nimbusml import Pipeline
from nimbusml.feature_extraction.text import NGramFeaturizer
from nimbusml.feature_extraction.text.extractor import Ngram
from nimbusml.feature_selection import MutualInformationSelector

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

X = train_reviews.loc[:, train_reviews.columns != 'like']
y = train_reviews['like']

# pipeline of transforms
transform_1 = NGramFeaturizer(word_feature_extractor=Ngram())
transform_2 = MutualInformationSelector(slots_in_output=2)
pipeline = Pipeline([transform_1, transform_2])
print(pipeline.fit_transform(X, y))

# Scikit compatibility (Compose transforms inside Scikit Pipeline).
# In this scenario, we do not provide {input, output} arguments
transform_1 = NGramFeaturizer(word_feature_extractor=Ngram())
transform_2 = MutualInformationSelector(slots_in_output=2)
pipe = Pipeline([
    ('text', transform_1),
    ('featureselect', transform_2)])
print(pipe.fit_transform(X, y))
