###############################################################################
# LightGbmClassifier
from nimbusml import Pipeline, FileDataStream
from nimbusml.datasets import get_dataset
from nimbusml.ensemble import LightGbmClassifier
from nimbusml.ensemble.booster import Dart
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
    LightGbmClassifier(feature=['parity', 'edu'], label='induced',
                       booster=Dart(reg_lambda=0.1))
])

# train, predict, and evaluate
# TODO: Replace with CV
metrics, predictions = pipeline.fit(data).test(data, output_scores=True)

# print predictions
print(predictions.head())
#   PredictedLabel   Score.0   Score.1   Score.2
# 0               2  0.070722  0.145439  0.783839
# 1               0  0.737733  0.260116  0.002150
# 2               2  0.070722  0.145439  0.783839
# 3               0  0.490715  0.091749  0.417537
# 4               0  0.562419  0.197818  0.239763
# print evaluation metrics
print(metrics)
#   Accuracy(micro-avg)  Accuracy(macro-avg)  Log-loss  Log-loss reduction  ...
# 0             0.641129             0.462618  0.772996          19.151269  ...
