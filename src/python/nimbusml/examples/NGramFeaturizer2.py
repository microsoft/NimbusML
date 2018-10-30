###############################################################################
# NGramFeaturizer
from nimbusml import FileDataStream
from nimbusml.datasets import get_dataset
from nimbusml.feature_extraction.text import NGramFeaturizer
from nimbusml.feature_extraction.text.extractor import Ngram
from nimbusml.feature_extraction.text.stopwords import CustomStopWordsRemover

# data input (as a FileDataStream)
path = get_dataset('wiki_detox_train').as_filepath()

data = FileDataStream.read_csv(path, sep='\t')
print(data.head())
#   Sentiment                                      SentimentText
# 0          1  ==RUDE== Dude, you are rude upload that carl p...
# 1          1  == OK! ==  IM GOING TO VANDALIZE WILD ONES WIK...
# 2          1  Stop trolling, zapatancas, calling me a liar m...
# 3          1  ==You're cool==  You seem like a really cool g...
# 4          1  ::::: Why are you threatening me? I'm not bein...

xf = NGramFeaturizer(word_feature_extractor=Ngram(),
                     stop_words_remover=CustomStopWordsRemover(['!',
                                                                '$',
                                                                '%',
                                                                '&',
                                                                '\'',
                                                                '\'d']),
                     columns={'features': ['SentimentText']})

# fit and transform
features = xf.fit_transform(data)

# print features
print(features.head())

#   Sentiment   ...         features.douchiest  features.award.
# 0          1  ...                        0.0              0.0
# 1          1  ...                        0.0              0.0
# 2          1  ...                        0.0              0.0
# 3          1  ...                        0.0              0.0
# 4          1  ...                        0.0              0.0
