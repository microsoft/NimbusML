###############################################################################
# OneVsRestClassifier
from nimbusml import Pipeline, FileDataStream
from nimbusml.datasets import get_dataset
from nimbusml.ensemble import FastTreesBinaryClassifier
from nimbusml.feature_extraction.categorical import OneHotVectorizer
from nimbusml.multiclass import OneVsRestClassifier

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
    OneVsRestClassifier(
        # using a binary classifier + OVR for multiclass dataset
        FastTreesBinaryClassifier(),
        # True = class probabilities will sum to 1.0
        # False = raw scores, unknown range
        use_probabilities=True,
        feature=['age', 'edu'], label='induced')
])

# train, predict, and evaluate
# TODO: Replace with CV
metrics, predictions = pipeline.fit(data).test(data, output_scores=True)

# print predictions
print(predictions.head())
#   PredictedLabel   Score.0   Score.1   Score.2
# 0               2  0.084504  0.302600  0.612897
# 1               0  0.620235  0.379226  0.000538
# 2               2  0.077734  0.061426  0.860840
# 3               0  0.657593  0.012318  0.330088
# 4               0  0.743427  0.090343  0.166231

# print evaluation metrics
print(metrics)
#   Accuracy(micro-avg)  Accuracy(macro-avg)  Log-loss  Log-loss reduction  ...
# 0             0.641129             0.515541  0.736198           22.99994  ...
