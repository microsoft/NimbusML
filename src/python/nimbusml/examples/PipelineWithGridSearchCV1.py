###############################################################################
# GridSearchCV with Pipeline: hyperparameter grid search.
import pandas as pd
from nimbusml import Pipeline
from nimbusml.ensemble import FastTreesBinaryClassifier
from nimbusml.feature_extraction.categorical import OneHotHashVectorizer, \
    OneHotVectorizer
from sklearn.model_selection import GridSearchCV

df = pd.DataFrame(dict(education=['A', 'A', 'A', 'A', 'B', 'A', 'B'],
                        workclass=['X', 'Y', 'X', 'X', 'X', 'Y', 'Y'],
                        y=[1, 0, 1, 1, 0, 1, 0]))
X = df.drop('y', axis=1)
y = df['y']
pipe = Pipeline([
    ('cat', OneHotVectorizer() << 'education'),
    # unnamed step, stays same in grid search
    OneHotHashVectorizer() << 'workclass',
    # this instance of FastTreesBinaryClassifier with number_of_trees 0 will be
    # never run by grid search as its not a part of param_grid below
    ('learner', FastTreesBinaryClassifier(number_of_trees=0, number_of_leaves=2))
])

param_grid = dict(
    cat__output_kind=[
        'Indicator', 'Binary'], learner__number_of_trees=[
        1, 2, 3])
grid = GridSearchCV(pipe, param_grid, cv=3, iid='warn')

grid.fit(X, y)
print(grid.best_params_)
# {'cat__output_kind': 'Indicator', 'learner__number_of_trees': 1}
