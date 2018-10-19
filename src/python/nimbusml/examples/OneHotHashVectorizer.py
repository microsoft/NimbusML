###############################################################################
# OneHotHashVectorizer
from nimbusml import FileDataStream
from nimbusml.datasets import get_dataset
from nimbusml.feature_extraction.categorical import OneHotHashVectorizer

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

xf = OneHotHashVectorizer(columns={'edu': 'education', 'sp': 'spontaneous'})

# fit and transform
features = xf.fit_transform(data)

print(features.head())
#    age  case  edu.0   edu.1003   ...    sp.995    ...   spontaneous  stratum
# 0    26     1    0.0        0.0   ...       0.0   ...           2.0      1.0
# 1    42     1    0.0        0.0   ...       0.0   ...           0.0      2.0
# 2    39     1    0.0        0.0   ...       0.0   ...           0.0      3.0
