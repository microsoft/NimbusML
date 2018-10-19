###############################################################################
# ColumnSelector
from nimbusml import FileDataStream
from nimbusml.datasets import get_dataset
from nimbusml.preprocessing.schema import ColumnSelector

# data input (as a FileDataStream)
path = get_dataset('infert').as_filepath()

data = FileDataStream.read_csv(path, sep=',')

# transform usage
xf = ColumnSelector(columns=['education', 'age'])

# fit and transform
features = xf.fit_transform(data)

# print features
print(features.head())
#   age education
# 0   26    0-5yrs
# 1   42    0-5yrs
# 2   39    0-5yrs
# 3   34    0-5yrs
# 4   35   6-11yrs
