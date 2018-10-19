###############################################################################
# BootstrapSampler
from nimbusml import FileDataStream
from nimbusml.datasets import get_dataset
from nimbusml.preprocessing.filter import BootstrapSampler

# data input (as a FileDataStream)
path = get_dataset('infert').as_filepath()

data = FileDataStream.read_csv(path, sep=',')

# transform usage
xf = BootstrapSampler()

# fit and transform
features = xf.fit_transform(data)

# print features
print(features.head())
#    age  case education  induced  parity  ... row_num  spontaneous     stratum
# 0   27     1   12+ yrs        1       3  ...      53            1          53
# 1   34     0   6-11yrs        0       2  ...     197            0          32
# 2   31     0   6-11yrs        1       2  ...     180            0          15
# 3   25     0   12+ yrs        1       1  ...     227            0          62
# 4   28     1   6-11yrs        0       2  ...      23            2          23
