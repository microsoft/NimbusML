###############################################################################
# AveragedPerceptronBinaryClassifier
from nimbusml import Pipeline, FileDataStream
from nimbusml.datasets import get_dataset
from nimbusml.linear_model import AveragedPerceptronBinaryClassifier

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
pipeline = Pipeline([AveragedPerceptronBinaryClassifier(
    feature=['age', 'parity', 'spontaneous'], label='case')])

# train, predict, and evaluate
# TODO: Replace with CV
metrics, predictions = pipeline.fit(data).test(data, output_scores=True)

# print predictions
print(predictions.head())
#   PredictedLabel     Score
# 0               0 -0.285667
# 1               0 -1.304729
# 2               0 -2.651955
# 3               0 -2.111450
# 4               0 -0.660658
# print evaluation metrics
print(metrics)
#        AUC  Accuracy  Positive precision  Positive recall  ...
# 0  0.705038   0.71371                 0.7         0.253012  ...
