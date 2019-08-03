###############################################################################
# EnsembleClassifier
from nimbusml import Pipeline, FileDataStream
from nimbusml.datasets import get_dataset
from nimbusml.feature_extraction.categorical import OneHotVectorizer
from nimbusml.ensemble import EnsembleClassifier
from nimbusml.ensemble.output_combiner import ClassifierVoting
from nimbusml.ensemble.sub_model_selector import ClassifierBestPerformanceSelector

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
                       sub_model_selector_type=ClassifierBestPerformanceSelector(),
                       output_combiner=ClassifierVoting())
])

# train, predict, and evaluate
# TODO: Replace with CV
metrics, predictions = pipeline.fit(data).test(data, output_scores=True)

# print predictions
print(predictions.head())
#    PredictedLabel  Score.0  Score.1  Score.2
# 0               2      0.0      0.0      1.0
# 1               0      1.0      0.0      0.0
# 2               2      0.0      0.0      1.0
# 3               0      1.0      0.0      0.0
# 4               0      1.0      0.0      0.0

# print evaluation metrics
print(metrics)
#    Accuracy(micro-avg)  Accuracy(macro-avg)   Log-loss  ...  (class 0)  ...
# 0             0.612903             0.417519  13.369849  ...   1.449179  ...
# (class 1)  (class 2)
# 29.967468  28.937894