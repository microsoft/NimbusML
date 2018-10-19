###############################################################################
# LogMeanVarianceScaler
import numpy
from nimbusml import FileDataStream
from nimbusml.datasets import get_dataset
from nimbusml.preprocessing.normalization import LogMeanVarianceScaler

# data input (as a FileDataStream)
path = get_dataset('infert').as_filepath()
data = FileDataStream.read_csv(
    path,
    sep=',',
    numeric_dtype=numpy.float32)  # Error with integer input
print(data.head())
#    age  case education  induced  parity  pooled.stratum  row_num  ...
# 0  26.0   1.0    0-5yrs      1.0     6.0             3.0      1.0  ...
# 1  42.0   1.0    0-5yrs      1.0     1.0             1.0      2.0  ...
# 2  39.0   1.0    0-5yrs      2.0     6.0             4.0      3.0  ...
# 3  34.0   1.0    0-5yrs      2.0     4.0             2.0      4.0  ...
# 4  35.0   1.0   6-11yrs      1.0     3.0            32.0      5.0  ...

# transform usage
xf = LogMeanVarianceScaler(columns={'in': 'induced', 'sp': 'spontaneous'})

# fit and transform
features = xf.fit_transform(data)

# print features
print(features.head())
#    age  case education        in  ...  row_num       sp  spontaneous  stratum
# 0  26.0   1.0    0-5yrs  0.230397 ...     1.0  0.919880          2.0      1.0
# 1  42.0   1.0    0-5yrs  0.230397 ...     2.0  0.000000          0.0      2.0
# 2  39.0   1.0    0-5yrs  0.912377 ...     3.0  0.000000          0.0      3.0
# 3  34.0   1.0    0-5yrs  0.912377 ...     4.0  0.000000          0.0      4.0
# 4  35.0   1.0   6-11yrs  0.230397 ...     5.0  0.238241          1.0      5.0
