###############################################################################
# ColumnConcatenator, ColumnDropper, ColumnSelector
import numpy as np
from nimbusml import Pipeline, Role
from nimbusml.datasets import get_dataset
from nimbusml.linear_model import LogisticRegressionClassifier
from nimbusml.preprocessing.schema import ColumnConcatenator
from nimbusml.preprocessing.schema import ColumnDropper
from nimbusml.preprocessing.schema import ColumnSelector
from sklearn.model_selection import train_test_split

# use 'iris' data set to create test and train data
#    Sepal_Length  Sepal_Width  Petal_Length  Petal_Width Label Species  Setosa
# 0           5.1          3.5           1.4          0.2     0  setosa     1.0
# 1           4.9          3.0           1.4          0.2     0  setosa     1.0
np.random.seed(0)
df = get_dataset("iris").as_df()

X_train, X_test, y_train, y_test = \
    train_test_split(df.loc[:, df.columns != 'Label'], df['Label'])

mycols = [
    'Sepal_Length',
    'Sepal_Width',
    'Petal_Length',
    'Petal_Width',
    'Setosa']

# drop 'Species' column using ColumnDropper Transform
# select mycols for training using ColumnConcatenator transform
dropcols = ColumnDropper() << 'Species'
concat = ColumnConcatenator() << {Role.Feature: mycols}

pipeline = Pipeline([dropcols, concat, LogisticRegressionClassifier()])
pipeline.fit(X_train, y_train)
scores1 = pipeline.predict(X_test)

# Select mycols using SelectColumns Transform
select = ColumnSelector() << mycols
pipeline.fit(X_train, y_train)
pipeline2 = Pipeline([select, LogisticRegressionClassifier()])
scores2 = pipeline.predict(X_test)

# Verify that we get identical results in both Experiments
print(scores1.head())
print(scores2.head())
