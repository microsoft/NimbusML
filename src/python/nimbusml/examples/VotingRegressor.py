from distutils.version import LooseVersion

import numpy as np
import pandas as pd

import sklearn
from sklearn.datasets import load_boston
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.ensemble import VotingRegressor as VotingRegressor_sklearn
from sklearn.model_selection import train_test_split

from nimbusml import Pipeline
from nimbusml.ensemble import VotingRegressor
from nimbusml.linear_model import (OrdinaryLeastSquaresRegressor,
                                   OnlineGradientDescentRegressor)


X, y = load_boston(return_X_y=True)
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=5, shuffle=False)

results = [('Expected Values', y_test)]


if LooseVersion(sklearn.__version__) >= LooseVersion('0.21'):

    # Perform regression using only scikit-learn classes
    reg1 = GradientBoostingRegressor(random_state=1, n_estimators=10)
    reg2 = LinearRegression()
    vr = VotingRegressor_sklearn(estimators=[('gb', reg1), ('rf', reg2)])
    vr.fit(X_train, y_train)
    result = vr.predict(X_test)
    results.append(('All scikit-learn', result))

    # Perform regression using the scikit-learn
    # VotingRegressor and NimbusML predictors.
    olsrArgs = { 'normalize': "Yes" }
    ogdArgs = {
        'shuffle': False,
        'number_of_iterations': 800,
        'learning_rate': 0.1,
        'normalize': "Yes"
    }
    r1 = OnlineGradientDescentRegressor(**ogdArgs)
    r2 = OrdinaryLeastSquaresRegressor(**olsrArgs)
    vr = VotingRegressor_sklearn(estimators=[('ogd', r1), ('ols', r2)])
    vr.fit(X_train, y_train)
    result = vr.predict(X_test)
    results.append(('scikit-learn VotingRegressor with NimbusML predictors', result))

# Perform regression using only NimbusML classes
olsrArgs = { 'normalize': "Yes" }
ogdArgs = {
    'shuffle': False,
    'number_of_iterations': 800,
    'learning_rate': 0.1,
    'normalize': "Yes"
}
r1 = OnlineGradientDescentRegressor(**ogdArgs)
r2 = OrdinaryLeastSquaresRegressor(**olsrArgs)
vr = VotingRegressor(estimators=[r1, r2], combiner='Average')
vr.fit(X_train, y_train)
result = vr.predict(X_test)
results.append(('All NimbusML', result))


print()
for result in results:
    print(result[0])
    print('-' * 50)
    print(result[1], end='\n\n\n')

# Expected Values
# --------------------------------------------------
# [22.4 20.6 23.9 22.  11.9]
# 
# 
# All scikit-learn
# --------------------------------------------------
# [23.4314038  22.85525696 28.02642909 25.62497493 22.84767784]
# 
# 
# scikit-learn VotingRegressor with NimbusML predictors
# --------------------------------------------------
# [24.873867 23.862251 29.125864 27.681105 24.006145]
# 
# 
# All NimbusML
# --------------------------------------------------
# 0    24.873867
# 1    23.862251
# 2    29.125864
# 3    27.681105
# 4    24.006145
# Name: Score, dtype: float32
