###############################################################################
# FastLinearClassifier
import numpy as np
import pandas as pd
from nimbusml.datasets import get_dataset
from nimbusml.linear_model import FastLinearClassifier
from sklearn.model_selection import train_test_split

# use 'iris' data set to create test and train data
#    Sepal_Length  Sepal_Width  Petal_Length  Petal_Width Label Species  Setosa
# 0           5.1          3.5           1.4          0.2     0  setosa     1.0
# 1           4.9          3.0           1.4          0.2     0  setosa     1.0
np.random.seed(0)

df = get_dataset("iris").as_df()
df.drop(['Species'], inplace=True, axis=1)

X_train, X_test, y_train, y_test = \
    train_test_split(df.loc[:, df.columns != 'Label'], df['Label'])
lr = FastLinearClassifier().fit(X_train, y_train)

scores = lr.predict(X_test)
scores = pd.to_numeric(scores)

# evaluate the model
print('Accuracy:', np.mean(y_test == [i for i in scores]))
