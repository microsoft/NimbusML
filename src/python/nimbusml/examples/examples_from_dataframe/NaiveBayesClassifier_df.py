###############################################################################
# NaiveBayesClassifier
import numpy as np
import pandas as pd
from nimbusml import Pipeline
from nimbusml.datasets import get_dataset
from nimbusml.feature_extraction.text import NGramFeaturizer
from nimbusml.feature_extraction.text.extractor import Ngram
from nimbusml.naive_bayes import NaiveBayesClassifier
from nimbusml.utils import get_X_y
from sklearn.model_selection import train_test_split

# use 'wiki_detox_train' data set to create test and train data
# Sentiment	SentimentText
# 1	  ==RUDE== Dude, you are rude upload that carl picture back, or else.
# 1	  == OK! ==  IM GOING TO VANDALIZE WILD ONES WIKI THEN!!!
np.random.seed(0)
train_file = get_dataset("wiki_detox_train").as_filepath()
(train, label) = get_X_y(train_file, label_column='Sentiment', sep='\t')

X_train, X_test, y_train, y_test = train_test_split(train, label)

# map text reviews to vector space
texttransform = NGramFeaturizer(
    word_feature_extractor=Ngram(),
    vector_normalizer='None') << 'SentimentText'
nb = NaiveBayesClassifier(feature=['SentimentText'])

ppl = Pipeline([texttransform, nb])
ppl.fit(X_train, y_train)

# evaluate the model
metrics, scores = ppl.test(X_test, y_test, output_scores=True)

print(metrics)
