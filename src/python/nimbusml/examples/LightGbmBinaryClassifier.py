###############################################################################
# LightGbmBinaryClassifier
from nimbusml import Pipeline, FileDataStream
from nimbusml.datasets import get_dataset
from nimbusml.ensemble import LightGbmBinaryClassifier
from nimbusml.ensemble.booster import Goss
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
    LightGbmBinaryClassifier(feature=['induced', 'edu'], label='case',
                             booster=Goss(top_rate=0.9))
])

# train, predict, and evaluate
# TODO: Replace with CV
metrics, predictions = pipeline.fit(
    data, 'case').test(
    data, output_scores=True)

# print predictions
print(predictions.head())
#   PredictedLabel  Probability     Score
# 0               1     0.612220  0.913309
# 1               1     0.612220  0.913309
# 2               0     0.334486 -1.375929
# 3               0     0.334486 -1.375929
# 4               0     0.421264 -0.635176
# print evaluation metrics
print(metrics)
#        AUC  Accuracy  Positive precision  Positive recall  ...
# 0  0.626433  0.677419            0.588235         0.120482  ...
