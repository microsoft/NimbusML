###############################################################################
# PcaAnomalyDetector
from nimbusml import Pipeline, FileDataStream
from nimbusml.datasets import get_dataset
from nimbusml.decomposition import PcaAnomalyDetector
from nimbusml.feature_extraction.categorical import OneHotVectorizer

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
    PcaAnomalyDetector(rank=3, feature=['induced', 'edu'])
])

# train, predict, and evaluate
# TODO: Replace with CV
metrics, predictions = pipeline.fit(data).test(
    data, 'case', output_scores=True)
#      Score
# 0  0.026155
# 1  0.026155
# 2  0.018055
# 3  0.018055
# 4  0.004043

# print predictions
print(predictions.head())

# print evaluation metrics
print(metrics)
#        AUC  DR @K FP  DR @P FPR  DR @NumPos  Threshold @K FP  ...
# 0  0.547718  0.084337          0    0.433735         0.009589  ...
