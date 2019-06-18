###############################################################################
# FromKey
from nimbusml import FileDataStream, Pipeline
from nimbusml.datasets import get_dataset
from nimbusml.preprocessing import FromKey, ToKey

# data input (as a FileDataStream)
path = get_dataset('topics').as_filepath()

# load data
data = FileDataStream.read_csv(path, sep=',')

# transform usage
pipeline = Pipeline([
    ToKey(columns=['review_reverse']),
    FromKey(columns=['review_reverse'])
])

# fit and transform
output = pipeline.fit_transform(data)
print(output.head())
#   label                              review                   review_reverse
# 0      1  animals birds cats dogs fish horse   radiation galaxy universe duck
# 1      0    horse birds house fish duck cats  space galaxy universe radiation
# 2      1         car truck driver bus pickup                       bus pickup
# 3      0   car truck driver bus pickup horse                        car truck
# 4      1     car truck  car truck driver bus                     pickup horse
