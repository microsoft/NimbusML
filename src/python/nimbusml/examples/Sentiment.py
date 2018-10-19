###############################################################################
# Sentiment: pre-trained transform to analyze sentiment of free-form
# text.
import pandas
from nimbusml.feature_extraction.text import Sentiment

# Create the data
customer_reviews = pandas.DataFrame(data=dict(review=[
    "I really did not like the taste of it",
    "It was surprisingly quite good!",
    "I will never ever ever go to that place again!!",
    "The best ever!! It was amazingly good and super fast",
    "I wish I had gone earlier, it was that great",
    "somewhat dissapointing. I'd probably wont try again",
    "Never visit again... rascals!"]))

analyze = Sentiment() << 'review'

# No need to fit any real data, just a dummy call to fit() to ensure the
# column name 'review' is present when transform() is invoked

# Skip until ML.NET resolve the resouce issue with Sentiment transform
# y = analyze.fit_transform(customer_reviews)

# View the sentiment scores!!
# print(y)
#     review
# 0  0.461790
# 1  0.960192
# 2  0.310344
# 3  0.965281
# 4  0.409665
# 5  0.492397
# 6  0.332535
