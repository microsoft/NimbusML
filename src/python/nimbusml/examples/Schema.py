# -*- coding: utf-8 -*-
###############################################################################
# Get schema from a fitted pipeline example.
import numpy as np
import pandas as pd
from nimbusml import Pipeline, FileDataStream
from nimbusml.datasets import get_dataset
from nimbusml.feature_extraction.text import NGramFeaturizer
from nimbusml.feature_extraction.text.extractor import Ngram

# data input (as a FileDataStream)
path = get_dataset("wiki_detox_train").as_filepath()

data = FileDataStream.read_csv(path, sep='\t')
print(data.head())
#    Sentiment                                      SentimentText
# 0          1  ==RUDE== Dude, you are rude upload that carl p...
# 1          1  == OK! ==  IM GOING TO VANDALIZE WILD ONES WIK...
# 2          1  Stop trolling, zapatancas, calling me a liar m...
# 3          1  ==You're cool==  You seem like a really cool g...
# 4          1  ::::: Why are you threatening me? I'm not bein...

pipe = Pipeline([
    NGramFeaturizer(
        word_feature_extractor=Ngram(),
        columns={
            'features': ['SentimentText']})
])

pipe.fit(data)
schema = pipe.get_output_columns()

print(schema[0:5])
# ['Sentiment', 'SentimentText', 'features.Char.<␂>|=|=', 'features.Char.=|=|r', 'features.Char.=|r|u']
