###############################################################################
# LpScaler
import numpy
from nimbusml import FileDataStream
from nimbusml.datasets import get_dataset
from nimbusml.preprocessing.normalization import LpScaler

path = get_dataset('infert').as_filepath()
data = FileDataStream.read_csv(
    path,
    sep=',',
    numeric_dtype=numpy.float32,
    collapse=True)

print(data.head())

# row_num education  age.age  age.parity  age.induced  age.case  age.spontaneous  age.stratum  age.pooled.stratum
#     1.0    0-5yrs     26.0         6.0          1.0       1.0              2.0          1.0                 3.0
#     2.0    0-5yrs     42.0         1.0          1.0       1.0              0.0          2.0                 1.0
#     3.0    0-5yrs     39.0         6.0          2.0       1.0              0.0          3.0                 4.0
#     4.0    0-5yrs     34.0         4.0          2.0       1.0              0.0          4.0                 2.0
#     5.0   6-11yrs     35.0         3.0          1.0       1.0              1.0          5.0                32.0

xf = LpScaler(columns={'norm': 'age'})
features = xf.fit_transform(data)

print_opts = {
    'index': False,
    'justify': 'left',
    'columns': [
        'norm.age',
        'norm.parity',
        'norm.induced',
        'norm.case',
        'norm.spontaneous',
        'norm.stratum',
        'norm.pooled.stratum'
    ]
}
print('LpScaler\n', features.head().to_string(**print_opts))

# norm.age  norm.parity  norm.induced  norm.case  norm.spontaneous  norm.stratum  norm.pooled.stratum
# 0.963624  0.222375     0.037062      0.037062   0.074125          0.037062      0.111187
# 0.997740  0.023756     0.023756      0.023756   0.000000          0.047511      0.023756
# 0.978985  0.150613     0.050204      0.025102   0.000000          0.075307      0.100409
# 0.982725  0.115615     0.057807      0.028904   0.000000          0.115615      0.057807
# 0.732032  0.062746     0.020915      0.020915   0.020915          0.104576      0.669286
