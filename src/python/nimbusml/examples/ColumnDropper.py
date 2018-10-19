###############################################################################
# ColumnDropper
import numpy
from nimbusml import FileDataStream
from nimbusml.datasets import get_dataset
from nimbusml.preprocessing.schema import ColumnDropper

# data input (as a FileDataStream)
path = get_dataset('infert').as_filepath()

data = FileDataStream.read_csv(path, sep=',', numeric_dtype=numpy.float32)

# transform usage
xf = ColumnDropper(columns=['education', 'age'])

# fit and transform
features = xf.fit_transform(data)

# print features
print(features.head())
#   case  induced  parity  pooled.stratum  row_num  spontaneous  stratum
# 0   1.0      1.0     6.0             3.0      1.0          2.0      1.0
# 1   1.0      1.0     1.0             1.0      2.0          0.0      2.0
# 2   1.0      2.0     6.0             4.0      3.0          0.0      3.0
# 3   1.0      2.0     4.0             2.0      4.0          0.0      4.0
# 4   1.0      1.0     3.0            32.0      5.0          1.0      5.0
