###############################################################################
# ColumnSelector
import numpy
from nimbusml import FileDataStream
from nimbusml.datasets import get_dataset
from nimbusml.preprocessing.normalization import GlobalContrastRowScaler

# data input (as a FileDataStream)
path = get_dataset('infert').as_filepath()
data = FileDataStream.read_csv(
    path,
    sep=',',
    numeric_dtype=numpy.float32,
    collapse=True)  # merge a few columns together
print(data.head())
#   age.age  age.case  age.induced  age.parity  age.pooled.stratum  ...
# 0     26.0       1.0          1.0         6.0                 3.0  ...
# 1     42.0       1.0          1.0         1.0                 1.0  ...
# 2     39.0       1.0          2.0         6.0                 4.0  ...
# 3     34.0       1.0          2.0         4.0                 2.0  ...
# 4     35.0       1.0          1.0         3.0                32.0  ...

xf = GlobalContrastRowScaler(columns={'ap': 'age'})

# fit and transform
features = xf.fit_transform(data)

# print features
print(features.head())
#   age.age  age.case  ... age.stratum    ap.age   ap.case  ...  ap.parity
# 0     26.0       1.0 ...         1.0  0.907723 -0.210950  ...   0.012785
# 1     42.0       1.0 ...         2.0  0.925178 -0.154196  ...   -0.154196
# 2     39.0       1.0 ...         3.0  0.916419 -0.201780  ...   -0.054649
