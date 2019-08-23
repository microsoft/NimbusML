###############################################################################
# DateTimeSplitter
from nimbusml import FileDataStream
from nimbusml.datasets import get_dataset
from nimbusml.preprocessing import DateTimeSplitter

# data input (as a FileDataStream)
path = get_dataset('infert').as_filepath()

data = FileDataStream.read_csv(path, sep=',')

# transform usage
xf = DateTimeSplitter(prefix='dt',
    columns={'education_copy': 'education'})

# fit and transform
features = xf.fit_transform(data)

# print features
print(features.head())
#   age  age_copy  case education education_copy  induced  parity  ...
# 0   26        26     1    0-5yrs         0-5yrs        1       6  ...
# 1   42        42     1    0-5yrs         0-5yrs        1       1  ...
# 2   39        39     1    0-5yrs         0-5yrs        2       6  ...
# 3   34        34     1    0-5yrs         0-5yrs        2       4  ...
# 4   35        35     1   6-11yrs        6-11yrs        1       3  ...
