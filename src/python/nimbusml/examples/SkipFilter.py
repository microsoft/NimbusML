import numpy as np
from nimbusml import FileDataStream
from nimbusml.datasets import get_dataset
from nimbusml.preprocessing.filter import SkipFilter, TakeFilter

# data input (as a FileDataStream)
path = get_dataset('infert').as_filepath()
data = FileDataStream.read_csv(
    path, sep=',', names={
        0: 'id'}, dtype={
        'id': str, 'age': np.float32})
print(data.head())
#    age  case education id  induced  parity  pooled.stratum  spontaneous  ...
# 0  26.0     1    0-5yrs  1        1       6               3            2  ...
# 1  42.0     1    0-5yrs  2        1       1               1            0  ...
# 2  39.0     1    0-5yrs  3        2       6               4            0  ...
# 3  34.0     1    0-5yrs  4        2       4               2            0  ...
# 4  35.0     1   6-11yrs  5        1       3              32            1  ...

# fit and transform
print(TakeFilter(count=100).fit_transform(data).shape)
# (100, 9), first 100 rows are preserved

print(SkipFilter(count=100).fit_transform(data).shape)
# (148, 9), first 100 rows are deleted
