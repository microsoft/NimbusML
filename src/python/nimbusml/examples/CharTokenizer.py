###############################################################################
# CharTokenizer
import numpy
from nimbusml import FileDataStream, DataSchema
from nimbusml.datasets import get_dataset

# data input (as a FileDataStream)
path = get_dataset('infert').as_filepath()

file_schema = DataSchema.read_schema(
    path, sep=',', numeric_dtype=numpy.float32)
data = FileDataStream(path, schema=file_schema)
print(data.schema)

# Section below throws "System.Runtime.InteropServices.SEHException"
# Logged in https://github.com/Microsoft/NimbusML/issues/31

# # transform usage
# xf = CharTokenizer(columns={'id_1': 'id', 'edu_1': 'education'})
#
# # fit and transform
# features = xf.fit_transform(data)
#
# # print features
# print(features.head())
