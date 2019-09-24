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
])

# fit with pandas dataframe
pipe.fit(X, Y)
schema = pipe.get_schema()
