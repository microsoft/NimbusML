###############################################################################
# LightLda
from nimbusml import FileDataStream, Pipeline
from nimbusml.datasets import get_dataset
from nimbusml.feature_extraction.text import NGramFeaturizer, LightLda
from nimbusml.feature_extraction.text.extractor import Ngram

# data input as a FileDataStream
path = get_dataset('topics').as_filepath()
data = FileDataStream.read_csv(path, sep=",")
print(data.head())
#                               review                    review_reverse  label
# 0  animals birds cats dogs fish horse   radiation galaxy universe duck      1
# 1    horse birds house fish duck cats  space galaxy universe radiation      0
# 2         car truck driver bus pickup                       bus pickup      1

# transform usage
pipeline = Pipeline(
    [
        NGramFeaturizer(
            word_feature_extractor=Ngram(),
            vector_normalizer='None',
            columns=['review']),
        LightLda(
            num_topic=3,
            columns=['review'])])

# fit and transform
features = pipeline.fit_transform(data)
print(features.head())
#   label  review.0  review.1  review.2                     review_reverse
# 0      1  0.500000  0.333333  0.166667     radiation galaxy universe duck
# 1      0  0.000000  0.166667  0.833333    space galaxy universe radiation
# 2      1  0.400000  0.200000  0.400000                         bus pickup
# 3      0  0.333333  0.333333  0.333333                          car truck
# 4      1  1.000000  0.000000  0.000000  car truck driver bus pickup horse
