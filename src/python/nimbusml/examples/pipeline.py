###############################################################################
# Pipeline
import numpy as np
import pandas as pd
from nimbusml import Pipeline, FileDataStream
from nimbusml.linear_model import FastLinearRegressor
from nimbusml.preprocessing.normalization import MeanVarianceScaler

X = np.array([[1, 2.0], [2, 4], [3, 0.7]])
Y = np.array([2, 3, 1.5])

df = pd.DataFrame(dict(y=Y, x1=X[:, 0], x2=X[:, 1]))

pipe = Pipeline([
    MeanVarianceScaler(),
    FastLinearRegressor()
])

# fit with pandas dataframe
pipe.fit(X, Y)

# Fit with FileDataStream
df.to_csv('data.csv', index=False)
ds = FileDataStream.read_csv('data.csv', sep=',')

pipe = Pipeline([
    MeanVarianceScaler(),
    FastLinearRegressor()
])
pipe.fit(ds, 'y')
print(pipe.summary())
#       Bias  Weights.x1  Weights.x2
# 0  1.032946    0.111758    1.210791
