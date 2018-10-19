###############################################################################
# OneHotVectorizer
from nimbusml import FileDataStream
from nimbusml.datasets import get_dataset
from nimbusml.feature_extraction.categorical import OneHotVectorizer

# data input (as a FileDataStream)
path = get_dataset('infert').as_filepath()
data = FileDataStream.read_csv(
    path, sep=',', dtype={
        'spontaneous': str})  # Error with numeric input for ohhv
print(data.head())
#    age  case education  induced  parity ... row_num  spontaneous  ...
# 0   26     1    0-5yrs        1       6 ...       1            2  ...
# 1   42     1    0-5yrs        1       1 ...       2            0  ...
# 2   39     1    0-5yrs        2       6 ...       3            0  ...
# 3   34     1    0-5yrs        2       4 ...       4            0  ...
# 4   35     1   6-11yrs        1       3 ...       5            1  ...

# transform usage
# set output_kind = "Bag" to featurize slots independently for vector type
# columns
xf = OneHotVectorizer(
    columns={
        'edu': 'education',
        'in': 'induced',
        'sp': 'spontaneous'})

# fit and transform
features = xf.fit_transform(data)
print(features.head())

#    age  case  edu.0-5yrs  edu.12+ yrs  edu.6-11yrs education   in.0  in.1 ...
# 0   26     1         1.0          0.0          0.0    0-5yrs    0.0   1.0 ...
# 1   42     1         1.0          0.0          0.0    0-5yrs    0.0   1.0 ...
# 2   39     1         1.0          0.0          0.0    0-5yrs    0.0   0.0 ...
