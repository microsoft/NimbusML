###############################################################################
# ToString

import pandas
from nimbusml.preprocessing import ToString, ToKey
from pandas import Categorical

# Create the data
categorical_df = pandas.DataFrame(data=dict(
    key=Categorical.from_codes([0, 1, 2, 1, 2, 0], categories=['a', 'b', 'c']),
    text=['b', 'c', 'a', 'b', 'a', 'c']))

print(categorical_df.dtypes)
# key     category
# text    object
# dtype: object

tostring = ToString(columns='key')
y = tostring.fit_transform(categorical_df)
print(y)
#   key text
# 0   1    b
# 1   2    c
# 2   3    a
# 3   2    b
# 4   3    a
# 5   1    c

print(y.dtypes)
# key     object  <== converted to string
# text    object
# dtype: object

tokey = ToKey(columns='text')
y = tokey.fit_transform(categorical_df)
y2 = tostring.clone().fit_transform(y)
print(y2['text'] == categorical_df['text'])
# 0    True
# 1    True
# 2    True
# 3    True
# 4    True
# 5    True
