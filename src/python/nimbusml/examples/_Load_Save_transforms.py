###############################################################################
# Example: pickle and unpickle a trained transform

import pickle

import numpy as np
from nimbusml.datasets import get_dataset
from nimbusml.feature_extraction.categorical import OneHotVectorizer

# Use the built-in data set 'infert' to create test and train data
#   Unnamed: 0  education   age  parity  induced  case  spontaneous  stratum  \
# 0           1        0.0  26.0     6.0      1.0   1.0          2.0      1.0
# 1           2        0.0  42.0     1.0      1.0   1.0          0.0      2.0
#   pooled.stratum education_str
# 0             3.0        0-5yrs
# 1             1.0        0-5yrs
np.random.seed(0)

df = get_dataset("infert").as_df()

# remove : and ' ' from column names, and encode categorical column
df.columns = [i.replace(': ', '') for i in df.columns]

cat = (OneHotVectorizer() << 'education_str').fit(df)
out1 = cat.transform(df)
print(out1.head())

# Unpickle transform and generate output.
# We should get the exact same output as above
s = pickle.dumps(cat)
cat2 = pickle.loads(s)
out2 = cat2.transform(df)
print(out1.head())
