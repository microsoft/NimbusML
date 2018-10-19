###############################################################################
# ColumnDuplicator
from nimbusml import FileDataStream
from nimbusml.datasets import get_dataset
from nimbusml.preprocessing.schema import ColumnDuplicator

# data input (as a FileDataStream)
path = get_dataset('infert').as_filepath()

data = FileDataStream.read_csv(path, sep=',')

# transform usage
xf = ColumnDuplicator(
    columns={
        'education_copy': 'education',
        'age_copy': 'age'})

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
