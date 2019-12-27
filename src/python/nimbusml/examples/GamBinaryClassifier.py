###############################################################################
# GamBinaryClassifier
from nimbusml import Pipeline, FileDataStream
from nimbusml.datasets import get_dataset
from nimbusml.ensemble import GamBinaryClassifier
from nimbusml.feature_extraction.categorical import OneHotVectorizer

# data input (as a FileDataStream)
path = get_dataset('infert').as_filepath()
data = FileDataStream.read_csv(path)
print(data.head())
#   age  case education  induced  parity  ... row_num  spontaneous  ...
# 0   26     1    0-5yrs        1       6 ...       1            2  ...
# 1   42     1    0-5yrs        1       1 ...       2            0  ...
# 2   39     1    0-5yrs        2       6 ...       3            0  ...
# 3   34     1    0-5yrs        2       4 ...       4            0  ...
# 4   35     1   6-11yrs        1       3 ...       5            1  ...

# define the training pipeline
pipeline = Pipeline([
    OneHotVectorizer(columns={'edu': 'education'}),
    GamBinaryClassifier(feature=['age', 'edu'], label='case')
])

# train, predict, and evaluate
metrics, predictions = pipeline.fit(data).test(data, output_scores=True)

# print predictions
print(predictions.head())
#   PredictedLabel     Score
# 0               0 -0.050461
# 1               0 -0.049737
# 2               0 -0.049737
# 3               0 -0.050461
# 4               0 -0.050552
# print evaluation metrics
print(metrics)
#        AUC  Accuracy  Positive precision  Positive recall  ...
# 0  0.502957  0.665323                   0                0  ...
