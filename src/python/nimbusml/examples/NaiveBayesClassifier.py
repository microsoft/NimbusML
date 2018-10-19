###############################################################################
# NaiveBayesClassifier
from nimbusml import Pipeline, FileDataStream
from nimbusml.datasets import get_dataset
from nimbusml.feature_extraction.categorical import OneHotVectorizer
from nimbusml.naive_bayes import NaiveBayesClassifier

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
    NaiveBayesClassifier(feature=['age', 'edu'], label='induced')
])

# train, predict, and evaluate
# TODO: Replace with CV
metrics, predictions = pipeline.fit(data).test(data, output_scores=True)

# print predictions
print(predictions.head())
#   PredictedLabel   Score.0   Score.1   Score.2
# 0               2 -5.297264 -5.873055 -4.847996
# 1               2 -5.297264 -5.873055 -4.847996
# 2               2 -5.297264 -5.873055 -4.847996
# 3               2 -5.297264 -5.873055 -4.847996
# 4               0 -1.785266 -3.172440 -3.691075

# print evaluation metrics
print(metrics)
#   Accuracy(micro-avg)  Accuracy(macro-avg)   Log-loss  Log-loss reduction ...
# 0             0.584677             0.378063  34.538776       -3512.460882 ...
