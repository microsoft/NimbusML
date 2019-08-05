###############################################################################
# EnsembleRegressor
from nimbusml import Pipeline, FileDataStream
from nimbusml.datasets import get_dataset
from nimbusml.feature_extraction.categorical import OneHotVectorizer
from nimbusml.ensemble import EnsembleRegressor
from nimbusml.ensemble.output_combiner import RegressorMedian
from nimbusml.ensemble.sub_model_selector import RegressorBestPerformanceSelector

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
    EnsembleRegressor(feature=['induced', 'edu'], label='age', num_models=3,
                      sub_model_selector_type=RegressorBestPerformanceSelector(),
                      output_combiner=RegressorMedian())
])

# train, predict, and evaluate
metrics, predictions = pipeline.fit(data).test(data, output_scores=True)

# print predictions
print(predictions.head())
#        Score
# 0  24.243475
# 1  24.243475
# 2  24.978357
# 3  24.978357
# 4  33.913101

# print evaluation metrics
print(metrics)
#     L1(avg)    L2(avg)  RMS(avg)  Loss-fn(avg)  R Squared
# 0  4.498702  31.549082  5.616857     31.549082  -0.148587

