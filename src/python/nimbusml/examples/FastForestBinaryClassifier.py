###############################################################################
# FastForestBinaryClassifier
import numpy
from nimbusml import Pipeline, FileDataStream
from nimbusml.datasets import get_dataset
from nimbusml.ensemble import FastForestBinaryClassifier
from nimbusml.feature_extraction.categorical import OneHotVectorizer

# data input (as a FileDataStream)
path = get_dataset('infert').as_filepath()
data = FileDataStream.read_csv(path, sep=',',
                               numeric_dtype=numpy.float32,
                               names={0: 'row_num', 5: 'case'})
print(data.head())
#    age  case education  induced  parity  pooled.stratum  row_num  ...
# 0  26.0   1.0    0-5yrs      1.0     6.0             3.0      1.0  ...
# 1  42.0   1.0    0-5yrs      1.0     1.0             1.0      2.0  ...
# 2  39.0   1.0    0-5yrs      2.0     6.0             4.0      3.0  ...
# 3  34.0   1.0    0-5yrs      2.0     4.0             2.0      4.0  ...
# 4  35.0   1.0   6-11yrs      1.0     3.0            32.0      5.0  ...
# define the training pipeline
pipeline = Pipeline([
    OneHotVectorizer(columns={'edu': 'education'}),
    FastForestBinaryClassifier(feature=['age', 'edu', 'induced'], label='case')
])

# train, predict, and evaluate
# TODO: Replace with CV
metrics, predictions = pipeline.fit(data).test(data, output_scores=True)

# print predictions
print(predictions.head())
#   PredictedLabel      Score
# 0             0.0 -26.985743
# 1             0.0 -26.562090
# 2             0.0 -24.832508
# 3             0.0 -23.799389
# 4             0.0 -19.612534
# print evaluation metrics
print(metrics)
#        AUC  Accuracy  Positive precision  Positive recall  ...
# 0  0.655714  0.665323                   0                0  ...
