###############################################################################
# LinearSvmBinaryClassifier
from nimbusml import Pipeline, FileDataStream
from nimbusml.datasets import get_dataset
from nimbusml.linear_model import LinearSvmBinaryClassifier

# data input (as a FileDataStream)
path = get_dataset('infert').as_filepath()

data = FileDataStream.read_csv(path)
print(data.head())
#   age  case education  induced  parity   ... row_num  spontaneous  ...
# 0   26     1    0-5yrs        1       6  ...       1            2  ...
# 1   42     1    0-5yrs        1       1  ...       2            0  ...
# 2   39     1    0-5yrs        2       6  ...       3            0  ...
# 3   34     1    0-5yrs        2       4  ...       4            0  ...
# 4   35     1   6-11yrs        1       3  ...       5            1  ...
# define the training pipeline
pipeline = Pipeline([LinearSvmBinaryClassifier(
    feature=['age', 'parity', 'spontaneous'], label='case')])

# train, predict, and evaluate
# TODO: Replace with CV
metrics, predictions = pipeline.fit(data).test(data, output_scores=True)

# print predictions
print(predictions.head())
#    PredictedLabel     Score  Probability
# 0               1  0.688481     0.607060
# 1               0 -2.514992     0.203312
# 2               0 -3.479344     0.129230
# 3               0 -3.016621     0.161422
# 4               0 -0.825512     0.397461
# print evaluation metrics
print(metrics)
#         AUC  Accuracy  Positive precision  Positive recall  ...
# 0  0.705476   0.71371            0.666667         0.289157  ...
