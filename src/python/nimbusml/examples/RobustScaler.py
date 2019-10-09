###############################################################################
# RobustScaler
import numpy
from nimbusml import FileDataStream
from nimbusml.datasets import get_dataset
from nimbusml.preprocessing.normalization import RobustScaler

# data input (as a FileDataStream)
path = get_dataset('infert').as_filepath()
data = FileDataStream.read_csv(path, sep=',')

print(data.head())
#    row_num education  age  parity  induced  case  spontaneous  stratum  pooled.stratum
# 0        1    0-5yrs   26       6        1     1            2        1               3
# 1        2    0-5yrs   42       1        1     1            0        2               1
# 2        3    0-5yrs   39       6        2     1            0        3               4
# 3        4    0-5yrs   34       4        2     1            0        4               2
# 4        5   6-11yrs   35       3        1     1            1        5              32

# transform usage
xf = RobustScaler(
    center=True, scale=True,
    columns={'age_norm': 'age', 'par_norm': 'parity'})

# fit and transform
features = xf.fit_transform(data)

print(features.head(n=10))
#    row_num education  age  parity  induced  case  spontaneous  stratum  pooled.stratum  age_norm  par_norm
# 0        1    0-5yrs   26       6        1     1            2        1               3 -0.434783       1.6
# 1        2    0-5yrs   42       1        1     1            0        2               1  0.956522      -0.4
# 2        3    0-5yrs   39       6        2     1            0        3               4  0.695652       1.6
# 3        4    0-5yrs   34       4        2     1            0        4               2  0.260870       0.8
# 4        5   6-11yrs   35       3        1     1            1        5              32  0.347826       0.4
# 5        6   6-11yrs   36       4        2     1            1        6              36  0.434783       0.8
# 6        7   6-11yrs   23       1        0     1            0        7               6 -0.695652      -0.4
# 7        8   6-11yrs   32       2        0     1            0        8              22  0.086957       0.0
# 8        9   6-11yrs   21       1        0     1            1        9               5 -0.869565      -0.4
# 9       10   6-11yrs   28       2        0     1            0       10              19 -0.260870       0.0
