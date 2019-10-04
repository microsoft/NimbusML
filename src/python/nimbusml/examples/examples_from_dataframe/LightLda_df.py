###############################################################################
# LightLda: cluster topics
import pandas
from nimbusml import Pipeline
from nimbusml.feature_extraction.text import LightLda, NGramFeaturizer
from nimbusml.feature_extraction.text.extractor import Ngram

# create the data
topics = pandas.DataFrame(data=dict(review=[
    "animals birds cats dogs fish horse",
    "horse birds house fish duck cats",
    "car truck driver bus pickup",
    "car truck driver bus pickup horse ",
    "car truck",
    "bus pickup",
    "space galaxy universe radiation",
    "radiation galaxy universe duck"]))

# there are three main topics in our data. set num_topic=3
# and see if LightLDA vectors for topics look similar
pipeline = Pipeline([NGramFeaturizer(word_feature_extractor=Ngram(
), vector_normalizer='None') << 'review', LightLda(num_topic=3)])
y = pipeline.fit_transform(topics)

# view the LDA topic vectors
print(y)
