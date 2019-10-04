###############################################################################
# WordTokenizer 

import pandas
from nimbusml import Pipeline
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

tokenize = WordTokenizer(char_array_term_separators=[" ", "n"]) << 'review'

pipeline = Pipeline([tokenize])

tokenize.fit(customer_reviews)
y = tokenize.transform(customer_reviews)

print(y)
#   review.00 review.01 review.02 review.03 review.04 review.05 review.06 review.07 review.08 review.09 review.10 review.11
# 0         I    really       did        ot      like       the     taste        of        it      None      None      None
# 1        It       was  surprisi       gly     quite     good!      None      None      None      None      None      None
# 2         I      will      ever      ever      ever        go        to      that     place      agai        !!      None
# 3       The      best    ever!!        It       was     amazi       gly      good         a         d     super      fast
# 4         I      wish         I       had        go         e  earlier,        it       was      that     great      None
# 5  somewhat  dissapoi        ti        g.       I'd  probably        wo         t       try      agai      None      None
# 6     Never     visit      agai       ...  rascals!      None      None      None      None      None      None      None
