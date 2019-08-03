###############################################################################
# EnsembleClassifier
from nimbusml import Pipeline, FileDataStream
from nimbusml.datasets import get_dataset
from nimbusml.feature_extraction.categorical import OneHotVectorizer
from nimbusml.ensemble import EnsembleClassifier
from nimbusml.ensemble.output_combiner import ClassifierVoting

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
    EnsembleClassifier(feature=['age', 'edu', 'parity'],
                       label='induced',
                       num_models=3,
                       output_combiner=ClassifierVoting())
])

# train, predict, and evaluate
# TODO: Replace with CV
metrics, predictions = pipeline.fit(data).test(data, output_scores=True)

# print predictions
print(predictions.head())
#    PredictedLabel   Score.0   Score.1   Score.2
# 0               2  0.202721  0.186598  0.628115
# 1               0  0.716737  0.190289  0.092974
# 2               2  0.201026  0.185602  0.624761
# 3               0  0.423328  0.235074  0.365649
# 4               0  0.577509  0.220827  0.201664

# print evaluation metrics
print(metrics)
#    Accuracy(micro-avg)  Accuracy(macro-avg)  Log-loss  ...  (class 0)  ...
# 0             0.612903             0.417519  0.846467  ...   0.504007  ...
# (class 1)  (class 2)
#  1.244033   1.439364
