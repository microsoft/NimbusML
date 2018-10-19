###############################################################################
# FastLinearClassifier
from nimbusml import Pipeline, FileDataStream
from nimbusml.datasets import get_dataset
from nimbusml.feature_extraction.categorical import OneHotVectorizer
from nimbusml.linear_model import FastLinearClassifier

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
    FastLinearClassifier(feature=['age', 'edu', 'parity'], label='induced')
])

# train, predict, and evaluate
# TODO: Replace with CV
metrics, predictions = pipeline.fit(data).test(data, output_scores=True)

# print predictions
print(predictions.head())
#   PredictedLabel   Score.0   Score.1   Score.2
# 0               2  0.015312  0.058199  0.926489
# 1               0  0.892915  0.097093  0.009991
# 2               2  0.058976  0.123581  0.817444
# 3               2  0.287882  0.245397  0.466721
# 4               0  0.404075  0.362293  0.233632

# print evaluation metrics
print(metrics)
#   Accuracy(micro-avg)  Accuracy(macro-avg)  Log-loss  Log-loss reduction  ...
# 0             0.604839             0.489888  0.857278
# 10.336077  ...
