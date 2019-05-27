###############################################################################
# FromKey

import pandas
from nimbusml.preprocessing import FromKey, ToKey
from pandas import Categorical

# Create the data
categorical_df = pandas.DataFrame(data=dict(
    key=Categorical.from_codes([0, 1, 2, 1, 2, 0], categories=['a', 'b', 'c']),
    text=['b', 'c', 'a', 'b', 'a', 'c']))

fromkey = FromKey(columns='key')
y = fromkey.fit_transform(categorical_df)
print(y)

tokey = ToKey(columns='text')
y = tokey.fit_transform(categorical_df)
y2 = fromkey.clone().fit_transform(y)
print(y2['text'] == categorical_df['text'])
