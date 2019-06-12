###############################################################################
# ColumnConcatenator
import numpy as np
import pandas as pd
from nimbusml import Pipeline, Role
from nimbusml.datasets import get_dataset
from nimbusml.linear_model import LogisticRegressionClassifier
from nimbusml.preprocessing.schema import ColumnConcatenator
from nimbusml.preprocessing.schema import ColumnDropper
from sklearn.model_selection import train_test_split

# use 'iris' data set to create test and train data
#    Sepal_Length  Sepal_Width  Petal_Length  Petal_Width Label Species  Setosa
# 0           5.1          3.5           1.4          0.2     0  setosa     1.0
# 1           4.9          3.0           1.4          0.2     0  setosa     1.0
df = get_dataset("iris").as_df()

X_train, X_test, y_train, y_test = \
    train_test_split(df.loc[:, df.columns != 'Label'], df['Label'])

dropcols = ColumnDropper() << 'Species'

mycols = [
    'Sepal_Length',
    'Sepal_Width',
    'Petal_Length',
    'Petal_Width',
    'Setosa']
concat = ColumnConcatenator() << {Role.Feature: mycols}

pipeline = Pipeline([dropcols, concat, LogisticRegressionClassifier()])
# TODO: fix as_matrix() requirement
pipeline.fit(X_train, y_train)

# Evaluate the model
metrics, scores = pipeline.test(X_test, y_test, output_scores=True)
print(metrics)
