###############################################################################
# WordTokenizer 

from nimbusml import Pipeline, FileDataStream
from nimbusml.datasets import get_dataset
from nimbusml.preprocessing.text import WordTokenizer

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

tokenize = WordTokenizer(char_array_term_separators=[" "]) << {'wt': 'SentimentText'}
pipeline = Pipeline([tokenize])

tokenize.fit(data)
y = tokenize.transform(data)

print(y.drop(labels='SentimentText', axis=1).head())
#    Sentiment    wt.000     wt.001       wt.002   wt.003       wt.004  wt.005  ... wt.366 wt.367 wt.368 wt.369 wt.370 wt.371 wt.372
# 0          1  ==RUDE==      Dude,          you      are         rude  upload  ...   None   None   None   None   None   None   None
# 1          1        ==        OK!           ==       IM        GOING      TO  ...   None   None   None   None   None   None   None
# 2          1      Stop  trolling,  zapatancas,  calling           me       a  ...   None   None   None   None   None   None   None
# 3          1  ==You're     cool==          You     seem         like       a  ...   None   None   None   None   None   None   None
# 4          1     :::::        Why          are      you  threatening     me?  ...   None   None   None   None   None   None   None
