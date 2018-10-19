###############################################################################
# Filter
import numpy as np
import pandas as pd
from nimbusml import FileDataStream
from nimbusml.preprocessing.missing_values import Handler

with_nans = pd.DataFrame(
    data=dict(
        Sepal_Length=[2.5, np.nan, 2.1, 1.0],
        Sepal_Width=[.75, .9, .8, .76],
        Petal_Length=[np.nan, 2.5, 2.6, 2.4],
        Petal_Width=[.8, .7, .9, 0.7],
        Species=["setosa", "viginica", "", 'versicolor']))

# write NaNs to file to show how this transform work
tmpfile = 'tmpfile_with_nans.csv'
with_nans.to_csv(tmpfile, index=False)

data = FileDataStream.read_csv(tmpfile, sep=',', numeric_dtype=np.float32)

# transform usage
xf = Handler(columns={'PL': 'Petal_Length'})

# fit and transform
features = xf.fit_transform(data)

# print features
print(features.head())

#   PL.IsMissing.Petal_Length  PL.Petal_Length  Petal_Length  Petal_Width  ...
# 0                        1.0              0.0           NaN          0.8  ...
# 1                        0.0              2.5           2.5          0.7  ...
# 2                        0.0              2.6           2.6          0.9  ...
# 3                        0.0              2.4           2.4          0.7  ...
