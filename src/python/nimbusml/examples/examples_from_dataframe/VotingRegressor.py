import numpy as np
import pandas as pd
from nimbusml import Pipeline
from nimbusml.ensemble import LightGbmRegressor, VotingRegressor
from nimbusml.linear_model import FastLinearRegressor, OrdinaryLeastSquaresRegressor

train_data = {'c1': [2, 3, 4, 5],
              'c2': [20, 30.8, 39.2, 51]}
train_df = pd.DataFrame(train_data).astype({'c1': np.float32,
                                            'c2': np.float32})

test_data = {'c1': [2.5, 3.5],
             'c2': [1, 1]}
test_df = pd.DataFrame(test_data).astype({'c1': np.float32,
                                          'c2': np.float32
                                          })

r1 = OrdinaryLeastSquaresRegressor(feature=['c1'], label='c2')
r1.fit(train_df)
result = r1.predict(test_df)
print(result)

r2 = FastLinearRegressor(feature=['c1'], label='c2', number_of_threads=1, shuffle=False)
r2.fit(train_df)
result = r2.predict(test_df)
print(result)

r3 = LightGbmRegressor(feature=['c1'], label='c2', random_state=1)
#r3.fit(train_df)
#result = r3.predict(test_df)
#print(result)


pipeline = Pipeline([VotingRegressor(estimators=[r1, r2])])
pipeline.fit(train_df)
result = pipeline.predict(test_df)
print(result)

