###############################################################################
# MutualInformationSelector
import numpy
from nimbusml import FileDataStream, Pipeline
from nimbusml.datasets import get_dataset
from nimbusml.feature_selection import MutualInformationSelector
from nimbusml.preprocessing.schema import ColumnConcatenator

# data input (as a FileDataStream)
path = get_dataset('infert').as_filepath()
data = FileDataStream.read_csv(
    path,
    sep=',',
    numeric_dtype=numpy.float32)  # Error with integer input
print(data.head())
#    age  case education  induced  parity  pooled.stratum  row_num  ...
# 0  26.0   1.0    0-5yrs      1.0     6.0             3.0      1.0  ...
# 1  42.0   1.0    0-5yrs      1.0     1.0             1.0      2.0  ...
# 2  39.0   1.0    0-5yrs      2.0     6.0             4.0      3.0  ...
# 3  34.0   1.0    0-5yrs      2.0     4.0             2.0      4.0  ...
# 4  35.0   1.0   6-11yrs      1.0     3.0            32.0      5.0  ...

# TODO: bug 244702
# As discussed, users can only use one column for this transform to select
# features from. We concatenate the target columns
# into one 'Feature' column with Vector type. The output, i.e. the selected
# features, will be named Features.*, and two slots
# are selected. We will support multiple columns in the future to
# concatenate columns automatically.
pip = Pipeline([
    ColumnConcatenator(
        columns={
            'Features': [
                'row_num',
                'spontaneous',
                'age',
                'parity',
                'induced']}),
    MutualInformationSelector(
        columns='Features',
        label='case',
        slots_in_output=2)  # only accept one column
])
pip.fit_transform(data).head()
#    Features.row_num  Features.spontaneous   age  case education  induced  ...
# 0               1.0                   2.0  26.0   1.0    0-5yrs      1.0  ...
# 1               2.0                   0.0  42.0   1.0    0-5yrs      1.0  ...
# 2               3.0                   0.0  39.0   1.0    0-5yrs      2.0  ...
# 3               4.0                   0.0  34.0   1.0    0-5yrs      2.0  ...
# 4               5.0                   1.0  35.0   1.0   6-11yrs      1.0  ...
