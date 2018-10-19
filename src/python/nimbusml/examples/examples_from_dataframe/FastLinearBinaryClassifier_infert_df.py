###############################################################################
# FastLinearBinaryClassifier
import numpy as np
from nimbusml.datasets import get_dataset
from nimbusml.feature_extraction.categorical import OneHotVectorizer
from nimbusml.linear_model import FastLinearBinaryClassifier
from sklearn.model_selection import train_test_split

# use the built-in data set 'infert' to create test and train data
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
df = (OneHotVectorizer() << 'education_str').fit_transform(df)

X_train, X_test, y_train, y_test = \
    train_test_split(df.loc[:, df.columns != 'case'], df['case'])

flinear = FastLinearBinaryClassifier().fit(X_train, y_train)
scores = flinear.predict(X_test)

# evaluate the model
print('Accuracy:', np.mean(y_test == [i for i in scores]))
