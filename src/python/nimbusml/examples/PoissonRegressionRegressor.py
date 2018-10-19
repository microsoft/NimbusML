###############################################################################
# PoissonRegressionRegressor
from nimbusml import Pipeline, FileDataStream
from nimbusml.datasets import get_dataset
from nimbusml.feature_extraction.categorical import OneHotVectorizer
from nimbusml.linear_model import PoissonRegressionRegressor

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
    PoissonRegressionRegressor(feature=['parity', 'edu'], label='age')
])

# train, predict, and evaluate
# TODO: Replace with CV
metrics, predictions = pipeline.fit(data).test(data, output_scores=True)

# print predictions
print(predictions.head())
#       Score
# 0  35.158913
# 1  35.191872
# 2  35.158913
# 3  35.172092
# 4  32.845158

# print evaluation metrics
print(metrics)
#    L1(avg)    L2(avg)  RMS(avg)  Loss-fn(avg)  R Squared
# 0  4.154053  24.429028  4.942573     24.429028   0.110628
