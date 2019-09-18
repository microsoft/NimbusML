###############################################################################
# ToKey
import numpy
from nimbusml import FileDataStream
from nimbusml.datasets import get_dataset
from nimbusml.preprocessing import ToString

# data input (as a FileDataStream)
path = get_dataset('infert').as_filepath()

data = FileDataStream.read_csv(path, sep=',', numeric_dtype=numpy.float32,
                               names={0: 'id'})
print(data.head())
#     id education   age  parity  induced  case  spontaneous  stratum  pooled.stratum
# 0  1.0    0-5yrs  26.0     6.0      1.0   1.0          2.0      1.0             3.0
# 1  2.0    0-5yrs  42.0     1.0      1.0   1.0          0.0      2.0             1.0
# 2  3.0    0-5yrs  39.0     6.0      2.0   1.0          0.0      3.0             4.0
# 3  4.0    0-5yrs  34.0     4.0      2.0   1.0          0.0      4.0             2.0
# 4  5.0   6-11yrs  35.0     3.0      1.0   1.0          1.0      5.0            32.0

# transform usage
xf = ToString(columns={'id_1': 'id', 'age_1': 'age'})

# fit and transform
features = xf.fit_transform(data)
print(features.head())
#     id education   age  parity  induced  case  spontaneous  stratum  pooled.stratum      id_1      age_1
# 0  1.0    0-5yrs  26.0     6.0      1.0   1.0          2.0      1.0             3.0  1.000000  26.000000
# 1  2.0    0-5yrs  42.0     1.0      1.0   1.0          0.0      2.0             1.0  2.000000  42.000000
# 2  3.0    0-5yrs  39.0     6.0      2.0   1.0          0.0      3.0             4.0  3.000000  39.000000
# 3  4.0    0-5yrs  34.0     4.0      2.0   1.0          0.0      4.0             2.0  4.000000  34.000000
# 4  5.0   6-11yrs  35.0     3.0      1.0   1.0          1.0      5.0            32.0  5.000000  35.000000

print(features.dtypes)
# id                float32
# education          object
# age               float32
# parity            float32
# induced           float32
# case              float32
# spontaneous       float32
# stratum           float32
# pooled.stratum    float32
# id_1               object  <== string column
# age_1              object  <== string column
