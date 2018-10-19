###############################################################################
# OneHotVectorizer
import pandas as pd
from nimbusml import Pipeline
from nimbusml.feature_extraction.categorical import OneHotVectorizer
from nimbusml.preprocessing.schema import ColumnConcatenator

data = pd.DataFrame({'month': ['Jan', 'Feb'], 'year': ['1988', '1978']})

# not concatenated
xf = OneHotVectorizer()
features = xf.fit_transform(data)
print(features.head())
#
#   month.Feb  month.Jan  year.1978  year.1988
# 0        0.0        1.0        0.0        1.0
# 1        1.0        0.0        1.0        0.0

# input columns concatenated into vector type
pipe = Pipeline([
    ColumnConcatenator(columns={'f': ['month', 'year']}),
    OneHotVectorizer(columns=['f']),
])
features2 = pipe.fit_transform(data)
print(features2.head())
#   f.month.1978  f.month.1988  f.month.Feb  f.month.Jan  f.year.1978  \
# 0           0.0           0.0          0.0          1.0          0.0
# 1           0.0           0.0          1.0          0.0          1.0
#
#   f.year.1988  f.year.Feb  f.year.Jan month  year
# 0          1.0         0.0         0.0   Jan  1988
# 1          0.0         0.0         0.0   Feb  1978

# input columns concatenated, output_kind = "Bag"
pipe = Pipeline([
    ColumnConcatenator(columns={'f': ['month', 'year']}),
    OneHotVectorizer(columns=['f'], output_kind="Bag"),
])
features2 = pipe.fit_transform(data)
print(features2.head())
#   f.1978  f.1988  f.Feb  f.Jan month  year
# 0     0.0     1.0    0.0    1.0   Jan  1988
# 1     1.0     0.0    1.0    0.0   Feb  1978
