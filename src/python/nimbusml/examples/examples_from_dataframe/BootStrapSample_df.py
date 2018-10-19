###############################################################################
# BootstrapSampler
# text.
import pandas
from nimbusml.preprocessing.filter import BootstrapSampler

# create the data
customer_reviews = pandas.DataFrame(data=dict(review=[
    "I really did not like the taste of it",
    "It was surprisingly quite good!",
    "I will never ever ever go to that place again!!",
    "The best ever!! It was amazingly good and super fast",
    "I wish I had gone earlier, it was that great",
    "somewhat dissapointing. I'd probably wont try again",
    "Never visit again... rascals!"]))

sample = BootstrapSampler(pool_size=20, complement=True)

y = sample.fit_transform(customer_reviews)

# view the sentiment scores!!
print(y)
