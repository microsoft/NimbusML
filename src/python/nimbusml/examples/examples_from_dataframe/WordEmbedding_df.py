###############################################################################
# WordEmbedding: pre-trained transform to generate word embeddings
import pandas
from nimbusml import Pipeline
from nimbusml.feature_extraction.text import WordEmbedding
from nimbusml.feature_extraction.text.ngramfeaturizer import NGramFeaturizer
from nimbusml.feature_extraction.text.extractor import Ngram

# create the data
customer_reviews = pandas.DataFrame(data=dict(review=[
    "I really did not like the taste of it",
    "It was surprisingly quite good!",
    "I will never ever ever go to that place again!!",
    "The best ever!! It was amazingly good and super fast",
    "I wish I had gone earlier, it was that great",
    "somewhat dissapointing. I'd probably wont try again",
    "Never visit again... rascals!"]))

pipeline = Pipeline([
    NGramFeaturizer(word_feature_extractor=Ngram(), output_tokens_column_name='review_TransformedText'),
    WordEmbedding() << 'review_TransformedText'
])
y = pipeline.fit_transform(customer_reviews)

# view the review embeddings
# print(y.head())
