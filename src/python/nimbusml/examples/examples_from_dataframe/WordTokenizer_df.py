###############################################################################
# WordTokenizer 

import pandas
from nimbusml import Pipeline, Role
from nimbusml.preprocessing.schema import ColumnConcatenator
from nimbusml.preprocessing.text import WordTokenizer

# create the data
customer_reviews = pandas.DataFrame(data=dict(review=[
    "I really did not like the taste of it",
    "It was surprisingly quite good!",
    "I will never ever ever go to that place again!!",
    "The best ever!! It was amazingly good and super fast",
    "I wish I had gone earlier, it was that great",
    "somewhat dissapointing. I'd probably wont try again",
    "Never visit again... rascals!"]))

tokenize = WordTokenizer(char_array_term_separators=[" ", "n"]) << ['review']

pipeline = Pipeline([tokenize])

tokenize.fit(customer_reviews)
y = tokenize.transform(customer_reviews)

print(y)
