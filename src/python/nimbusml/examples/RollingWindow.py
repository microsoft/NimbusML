###############################################################################
# DateTimeSplitter
import pandas as pd
import numpy as np
from nimbusml import FileDataStream
from nimbusml.datasets import get_dataset
from nimbusml.timeseries import RollingWindow

# data input (as a FileDataStream)
path = get_dataset('infert').as_filepath()
data = FileDataStream.read_csv(path, sep=',', numeric_dtype=np.double)

# transform usage
xf = RollingWindow(columns={'age_1': 'age'},
                   grain_column=['education'], 
                   window_calculation='Mean',
                   max_window_size=2,
                   horizon=2)

# fit and transform
features = xf.fit_transform(data)

features = features.drop(['row_num', 'education', 'parity', 'induced',
                          'case', 'spontaneous', 'stratum', 'pooled.stratum'], axis=1)

# print features
print(features.head(100))
