# -*- coding: utf-8 -*-
###############################################################################
# NGramExtractor
from nimbusml import FileDataStream, Pipeline
from nimbusml.datasets import get_dataset
from nimbusml.preprocessing.schema import ColumnDropper
from nimbusml.preprocessing.text import CharTokenizer
from nimbusml.feature_extraction.text import NGramExtractor

# data input (as a FileDataStream)
path = get_dataset("wiki_detox_train").as_filepath()

data = FileDataStream.read_csv(path, sep='\t')
print(data.head())
#   Sentiment                                      SentimentText
# 0          1  ==RUDE== Dude, you are rude upload that carl p...
# 1          1  == OK! ==  IM GOING TO VANDALIZE WILD ONES WIK...
# 2          1  Stop trolling, zapatancas, calling me a liar m...
# 3          1  ==You're cool==  You seem like a really cool g...
# 4          1  ::::: Why are you threatening me? I'm not bein...

# transform usage
pipe = Pipeline([
        CharTokenizer(columns={'SentimentText_Transform': 'SentimentText'}),
        NGramExtractor(ngram_length=1, all_lengths=False, columns={'Ngrams': 'SentimentText_Transform'}),
        ColumnDropper(columns=['SentimentText_Transform', 'SentimentText', 'Sentiment'])
        ])

# fit and transform
features = pipe.fit_transform(data)

print(features.head())
#    Ngrams.<␂>  Ngrams.=  Ngrams.R  Ngrams.U  Ngrams.D  Ngrams.E  ...
# 0         1.0       4.0       1.0       1.0       2.0       1.0  ...
# 1         1.0       4.0       0.0       0.0       2.0       3.0  ...
# 2         1.0       0.0       0.0       0.0       0.0       0.0  ...
# 3         1.0       4.0       0.0       0.0       0.0       0.0  ...
# 4         1.0       0.0       0.0       0.0       0.0       0.0  ...
