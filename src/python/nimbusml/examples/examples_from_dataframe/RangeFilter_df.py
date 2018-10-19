###############################################################################
# RangeFilter
import numpy as np
from nimbusml.datasets import get_dataset
from nimbusml.preprocessing.filter import RangeFilter
from sklearn.model_selection import train_test_split

# use 'iris' data set to create test and train data
#    Sepal_Length  Sepal_Width  Petal_Length  Petal_Width Label Species  Setosa
# 0           5.1          3.5           1.4          0.2     0  setosa     1.0
# 1           4.9          3.0           1.4          0.2     0  setosa     1.0
np.random.seed(0)
df = get_dataset("iris").as_df()

X_train, X_test, y_train, y_test = \
    train_test_split(df.loc[:, df.columns != 'Label'], df['Label'])

# select rows where  5.0 <= Sepal_Length <= 5.1
filter = RangeFilter(min=5.0, max=5.1) << 'Sepal_Length'
print(filter.fit_transform(X_train))

# select rows where Sepal_Length <= 4.5 or Sepal_Length >= 7.5
filter = RangeFilter(min=4.5, max=7.5, complement=True) << 'Sepal_Length'
