###############################################################################
# WordEmbedding: pre-trained transform to generate word embeddings
import pandas
from nimbusml import Pipeline
from nimbusml.feature_extraction.text import NGramFeaturizer, WordEmbedding
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

# view a small subset of the review embeddings
print(y.iloc[:5, -3:])
#    review_TransformedText.147  review_TransformedText.148  review_TransformedText.149
# 0                    1.918661                   -0.714531                    3.062141
# 1                    1.891922                   -0.248650                    1.706620
# 2                    1.601611                    0.309785                    3.379576
# 3                    1.970666                    1.477450                    3.110802
# 4                    2.521791                    0.122538                    3.129919

