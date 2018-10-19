###############################################################################
# Example: pickle and unpickle transforms and trainers in a pipeline

import pickle

import numpy as np
from nimbusml.datasets import get_dataset
from nimbusml.ensemble import FastTreesBinaryClassifier
from nimbusml.feature_extraction.categorical import OneHotVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Use the built-in data set 'infert' to create test and train data
#   Unnamed: 0  education   age  parity  induced  case  spontaneous  stratum  \
# 0           1        0.0  26.0     6.0      1.0   1.0          2.0      1.0
# 1           2        0.0  42.0     1.0      1.0   1.0          0.0      2.0
#   pooled.stratum education_str
# 0             3.0        0-5yrs
# 1             1.0        0-5yrs
np.random.seed(0)

df = get_dataset("infert").as_df()

# remove : and ' ' from column names
df.columns = [i.replace(': ', '') for i in df.columns]

X_train, X_test, y_train, y_test = \
    train_test_split(df.loc[:, df.columns != 'case'], df['case'])

# pipeline with 1 transform and a trainer
cat = OneHotVectorizer() << 'education_str'
ftree = FastTreesBinaryClassifier()
pipe = Pipeline(steps=[("cat", cat), ("ftree", ftree)])
pipe.fit(X_train, y_train.to_frame())

scores = pipe.predict(X_test)
print('Accuracy1:', np.mean(y_test == [i for i in scores]))

# Unpickle model and score. We should get the exact same accuracy as above
s = pickle.dumps(pipe)
pipe2 = pickle.loads(s)
scores2 = pipe2.predict(X_test)
print('Accuracy2:', np.mean(y_test == [i for i in scores2]))
