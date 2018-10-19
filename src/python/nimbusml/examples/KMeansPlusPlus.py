###############################################################################
# KMeansPlusPlus
from nimbusml import Pipeline, FileDataStream
from nimbusml.cluster import KMeansPlusPlus
from nimbusml.datasets import get_dataset
from nimbusml.feature_extraction.categorical import OneHotVectorizer

# data input (as a FileDataStream)
path = get_dataset('infert').as_filepath()

data = FileDataStream.read_csv(path)
print(data.head())
#    age  case education  induced  parity ... row_num  spontaneous  ...
# 0   26     1    0-5yrs        1       6 ...       1            2  ...
# 1   42     1    0-5yrs        1       1 ...       2            0  ...
# 2   39     1    0-5yrs        2       6 ...       3            0  ...
# 3   34     1    0-5yrs        2       4 ...       4            0  ...
# 4   35     1   6-11yrs        1       3 ...       5            1  ...

# define the training pipeline
pipeline = Pipeline([
    OneHotVectorizer(columns={'edu': 'education'}),
    KMeansPlusPlus(n_clusters=5, feature=['induced', 'edu', 'parity'])
])

# train, predict, and evaluate
# TODO: Replace with CV
metrics, predictions = pipeline \
    .fit(data) \
    .test(data, 'induced', output_scores=True)

# print predictions
print(predictions.head())
#   PredictedLabel   Score.0   Score.1   Score.2   Score.3   Score.4
# 0               4  2.732253  2.667988  2.353899  2.339244  0.092014
# 1               4  2.269290  2.120064  2.102576  2.222578  0.300347
# 2               4  3.482253  3.253153  2.425328  2.269245  0.258680
# 3               4  3.130401  2.867317  2.158132  2.055911  0.175347
# 4               2  0.287809  2.172567  0.036439  2.102578  2.050347

# print evaluation metrics
print(metrics)
#        NMI  AvgMinScore
# 0  0.521177     0.074859
