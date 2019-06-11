###############################################################################
# GridSearchCV with Pipeline: grid search over learners
import pandas as pd
from nimbusml import Pipeline
from nimbusml.ensemble import FastTreesBinaryClassifier, GamBinaryClassifier
from nimbusml.feature_extraction.categorical import OneHotHashVectorizer
from nimbusml.linear_model import FastLinearBinaryClassifier, \
    LogisticRegressionBinaryClassifier
from sklearn.model_selection import GridSearchCV

df = pd.DataFrame(dict(education=['A', 'A', 'A', 'A', 'B', 'A', 'B'],
                        workclass=['X', 'Y', 'X', 'X', 'X', 'Y', 'Y'],
                        y=[1, 0, 1, 1, 0, 1, 0]))
X = df.drop('y', axis=1)
y = df['y']

cat = OneHotHashVectorizer() << ['education', 'workclass']
learner = FastTreesBinaryClassifier()
pipe = Pipeline(steps=[('cat', cat), ('learner', learner)])

param_grid = dict(cat__number_of_bits=[1, 2, 4, 6, 8, 16],
                  learner=[
                      FastLinearBinaryClassifier(),
                      FastTreesBinaryClassifier(),
                      LogisticRegressionBinaryClassifier(),
                      GamBinaryClassifier()
                  ])
grid = GridSearchCV(pipe, param_grid, cv=3, iid='warn', )

grid.fit(X, y)
print(grid.best_params_['learner'].__class__.__name__)
# FastLinearBinaryClassifier
print(grid.best_params_['cat__number_of_bits'])
# 2
