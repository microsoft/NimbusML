###############################################################################
# ToKey
import numpy
from nimbusml import FileDataStream
from nimbusml.datasets import get_dataset
from nimbusml.preprocessing import ToKeyImputer

# data input (as a FileDataStream)
path = get_dataset('airquality').as_filepath()

data = FileDataStream.read_csv(path, sep=',', numeric_dtype=numpy.float32,
                               names={0: 'id'})
print(data.head(6))
#     id  Ozone  Solar_R  Wind  Temp  Month  Day
# 0  1.0   41.0    190.0   7.4  67.0    5.0  1.0
# 1  2.0   36.0    118.0   8.0  72.0    5.0  2.0
# 2  3.0   12.0    149.0  12.6  74.0    5.0  3.0
# 3  4.0   18.0    313.0  11.5  62.0    5.0  4.0
# 4  5.0    NaN      NaN  14.3  56.0    5.0  5.0
# 5  6.0   28.0      NaN  14.9  66.0    5.0  6.0


# transform usage
xf = ToKeyImputer(columns={'Ozone_1': 'Ozone', 'Solar_R_1': 'Solar_R'})

# fit and transform
features = xf.fit_transform(data)
print(features.head(6))
#     id  Ozone  Solar_R  Wind  Temp  Month  Day  Ozone_1  Solar_R_1
# 0  1.0   41.0    190.0   7.4  67.0    5.0  1.0     41.0      190.0
# 1  2.0   36.0    118.0   8.0  72.0    5.0  2.0     36.0      118.0
# 2  3.0   12.0    149.0  12.6  74.0    5.0  3.0     12.0      149.0
# 3  4.0   18.0    313.0  11.5  62.0    5.0  4.0     18.0      313.0
# 4  5.0    NaN      NaN  14.3  56.0    5.0  5.0     23.0      238.0   <== Missing values have been updated
# 5  6.0   28.0      NaN  14.9  66.0    5.0  6.0     28.0      238.0   <== Missing values have been updated
