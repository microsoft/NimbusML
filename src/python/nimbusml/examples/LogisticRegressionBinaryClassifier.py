###############################################################################
# LogisticRegressionBinaryClassifier
from nimbusml import Pipeline, FileDataStream
from nimbusml.datasets import get_dataset
from nimbusml.feature_extraction.categorical import OneHotVectorizer
from nimbusml.linear_model import LogisticRegressionBinaryClassifier

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
    LogisticRegressionBinaryClassifier(feature=['parity', 'edu'], label='case')
])

# train, predict, and evaluate
# TODO: Replace with CV
metrics, predictions = pipeline.fit(data).test(data, output_scores=True)

# print predictions
print(predictions.head())
#   PredictedLabel  Probability     Score
# 0               0     0.334679 -0.687098
# 1               0     0.334679 -0.687098
# 2               0     0.334679 -0.687098
# 3               0     0.334679 -0.687098
# 4               0     0.334679 -0.687098
# print evaluation metrics
print(metrics)
#   AUC  Accuracy  Positive precision  Positive recall  Negative precision  ...
# 0  0.5  0.665323                   0                0           0.665323  ...
