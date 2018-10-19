###############################################################################
# Filter
import numpy as np
import pandas as pd
from nimbusml import FileDataStream
from nimbusml.preprocessing.missing_values import Filter

with_nans = pd.DataFrame(
    data=dict(
        Sepal_Length=[2.5, np.nan, 2.1, 1.0],
        Sepal_Width=[.75, .9, .8, .76],
        Petal_Length=[np.nan, 2.5, 2.6, 2.4],
        Petal_Width=[.8, .7, .9, 0.7]))

# write NaNs to file to show how this transform work
tmpfile = 'tmpfile_with_nans.csv'
with_nans.to_csv(tmpfile, index=False)

data = FileDataStream.read_csv(tmpfile, sep=',', numeric_dtype=np.float32)

# transform usage
xf = Filter(
    columns=[
        'Petal_Length',
        'Petal_Width',
        'Sepal_Length',
        'Sepal_Width'])

# fit and transform
features = xf.fit_transform(data)

# print features
print(features.head())
#    Petal_Length  Petal_Width  Sepal_Length  Sepal_Width
# 0           2.4          0.7           1.0         0.76
