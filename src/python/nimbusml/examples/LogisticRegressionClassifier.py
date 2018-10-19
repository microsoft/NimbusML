###############################################################################
# LogisticRegressionClassifier
from nimbusml import Pipeline, FileDataStream
from nimbusml.datasets import get_dataset
from nimbusml.feature_extraction.categorical import OneHotVectorizer
from nimbusml.linear_model import LogisticRegressionClassifier

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
    LogisticRegressionClassifier(feature=['parity', 'edu'], label='induced')
])

# train, predict, and evaluate
# TODO: Replace with CV
metrics, predictions = pipeline.fit(data).test(data, output_scores=True)

# print predictions
print(predictions.head())
#   PredictedLabel   Score.0   Score.1   Score.2
# 0               2  0.171122  0.250151  0.578727
# 1               0  0.678313  0.220665  0.101022
# 2               2  0.171122  0.250151  0.578727
# 3               0  0.360849  0.289190  0.349961
# 4               0  0.556921  0.260420  0.182658
# print evaluation metrics
print(metrics)
#   Accuracy(micro-avg)  Accuracy(macro-avg)  Log-loss  Log-loss reduction  ...
# 0             0.592742             0.389403  0.857392          10.324157  ...
