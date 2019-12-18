import numpy as np
import pandas as pd
from nimbusml import Pipeline
from nimbusml.ensemble import LightGbmRegressor, VotingRegressor
from nimbusml.linear_model import (OrdinaryLeastSquaresRegressor,
                                   OnlineGradientDescentRegressor)

show_individual_predictions = False

np.random.seed(0)
x = np.arange(1000, step=0.1)
y = x * 10 + (np.random.standard_normal(len(x)) * 10)
train_data = {'c1': x, 'c2': y}
train_df = pd.DataFrame(train_data).astype({'c1': np.float32, 'c2': np.float32})

test_data = {'c1': [2.5, 30.5], 'c2': [1, 1]}
test_df = pd.DataFrame(test_data).astype({'c1': np.float32, 'c2': np.float32})

olsrArgs = {
    'feature': ['c1'],
    'label': 'c2',
    'normalize': "Yes"
}

ogdArgs = {
    'feature': ['c1'],
    'label': 'c2',
    'shuffle': False,
    'number_of_iterations': 800,
    'learning_rate': 0.1,
    'normalize': "Yes"
}

lgbmArgs = {
    'feature':['c1'],
    'label':'c2',
    'random_state':1,
    'number_of_leaves':200,
    'minimum_example_count_per_leaf': 1,
    'normalize': 'Yes'
}

if show_individual_predictions:
    r1 = OrdinaryLeastSquaresRegressor(**olsrArgs)
    r1.fit(train_df)
    result = r1.predict(test_df)
    print(result)

    r2 = OnlineGradientDescentRegressor(**ogdArgs)
    r2.fit(train_df)
    result = r2.predict(test_df)
    print(result)

    r3 = LightGbmRegressor(**lgbmArgs)
    r3.fit(train_df)
    result = r3.predict(test_df)
    print(result)

# Perform a prediction using an ensemble
# of all three of the above predictors.

r1 = OrdinaryLeastSquaresRegressor(**olsrArgs)
r2 = OnlineGradientDescentRegressor(**ogdArgs)
r3 = LightGbmRegressor(**lgbmArgs)
pipeline = Pipeline([VotingRegressor(estimators=[r1, r2, r3], combiner='Average')])

pipeline.fit(train_df)
result = pipeline.predict(test_df)
print(result)

