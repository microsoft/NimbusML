###############################################################################
# CharTokenizer
import numpy
from nimbusml import FileDataStream, DataSchema, Pipeline
from nimbusml.datasets import get_dataset
from nimbusml.preprocessing import FromKey
from nimbusml.preprocessing.text import CharTokenizer
from nimbusml.preprocessing.schema import ColumnSelector
from nimbusml.feature_extraction.text import WordEmbedding

# data input (as a FileDataStream)
path = get_dataset('wiki_detox_train').as_filepath()

file_schema = DataSchema.read_schema(
    path, sep='\t', numeric_dtype=numpy.float32)
data = FileDataStream(path, schema=file_schema)
print(data.head())

#    Sentiment                                      SentimentText
# 0        1.0  ==RUDE== Dude, you are rude upload that carl p...
# 1        1.0  == OK! ==  IM GOING TO VANDALIZE WILD ONES WIK...
# 2        1.0  Stop trolling, zapatancas, calling me a liar m...
# 3        1.0  ==You're cool==  You seem like a really cool g...
# 4        1.0  ::::: Why are you threatening me? I'm not bein...

# After using Character Tokenizer, it will convert the vector of Char to Key type.
# Use FromKey to retrieve the data from Key first, then send into WordEmbedding.

pipe = Pipeline([
        CharTokenizer(columns={'SentimentText_Transform': 'SentimentText'}),
        FromKey(columns={'SentimentText_FromKey': 'SentimentText_Transform'}),
        WordEmbedding(model_kind='GloVe50D', columns={'Feature': 'SentimentText_FromKey'}),
        ColumnSelector(columns=['Sentiment', 'SentimentText', 'Feature'])
        ])

print(pipe.fit_transform(data).head())

#    Sentiment  ... Feature.149
# 0        1.0  ...     2.67440
# 1        1.0  ...     0.78858
# 2        1.0  ...     2.67440
# 3        1.0  ...     2.67440
# 4        1.0  ...     2.67440

# [5 rows x 152 columns]
