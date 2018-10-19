###############################################################################
# PcaTransformer
import numpy
from nimbusml import FileDataStream
from nimbusml.datasets import get_dataset
from nimbusml.decomposition import PcaTransformer

# data input (as a FileDataStream)
path = get_dataset('infert').as_filepath()

data = FileDataStream.read_csv(path, sep=',', numeric_dtype=numpy.float32)

# transform data
feature_columns = ['age', 'parity', 'induced', 'spontaneous']

pipe = PcaTransformer(rank=3, columns={'features': feature_columns})

print(pipe.fit_transform(data).head())
#     age  case education  features.0  features.1  features.2  induced  ...
# 0  26.0   1.0    0-5yrs   -5.675901   -3.964389   -1.031570      1.0  ...
# 1  42.0   1.0    0-5yrs   10.364552    0.875251    0.773911      1.0  ...
# 2  39.0   1.0    0-5yrs    7.336117   -4.073389    1.128798      2.0  ...
# 3  34.0   1.0    0-5yrs    2.340584   -2.130528    1.248973      2.0  ...
# 4  35.0   1.0   6-11yrs    3.343876   -1.088401   -0.100063      1.0  ...
