###############################################################################
# EnsembleRegressor
from nimbusml import Pipeline, FileDataStream
from nimbusml.datasets import get_dataset
from nimbusml.feature_extraction.categorical import OneHotVectorizer
from nimbusml.ensemble import EnsembleRegressor
from nimbusml.ensemble.feature_selector import RandomFeatureSelector
from nimbusml.ensemble.output_combiner import RegressorMedian
from nimbusml.ensemble.subset_selector import RandomPartitionSelector
from nimbusml.ensemble.sub_model_selector import RegressorBestDiverseSelector

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

# define the training pipeline using default sampling and ensembling parameters
pipeline_with_defaults = Pipeline([
    OneHotVectorizer(columns={'edu': 'education'}),
    EnsembleRegressor(feature=['induced', 'edu'], label='age', num_models=3)
])

# train, predict, and evaluate
metrics, predictions = pipeline_with_defaults.fit(data).test(data, output_scores=True)

# print predictions
print(predictions.head())
#        Score
# 0  26.046741
# 1  26.046741
# 2  29.225840
# 3  29.225840
# 4  33.849384

# print evaluation metrics
print(metrics)
#    L1(avg)    L2(avg)  RMS(avg)  Loss-fn(avg)  R Squared
# 0  4.69884  33.346123   5.77461     33.346124  -0.214011


# define the training pipeline with specific sampling and ensembling options
pipeline_with_options = Pipeline([
    OneHotVectorizer(columns={'edu': 'education'}),
    EnsembleRegressor(feature=['induced', 'edu'],
                      label='age',
                      num_models=3,
                      sampling_type = RandomPartitionSelector(
                          feature_selector=RandomFeatureSelector(
                               features_selction_proportion=0.7)),
                       sub_model_selector_type=RegressorBestDiverseSelector(),
                       output_combiner=RegressorMedian())
])

# train, predict, and evaluate
metrics, predictions = pipeline_with_options.fit(data).test(data, output_scores=True)

# print predictions
print(predictions.head())
#        Score
# 0  37.122200
# 1  37.122200
# 2  41.296204
# 3  41.296204
# 4  33.591423

# print evaluation metrics
# note that the converged loss function values are worse than with defaults as
# this is a small dataset that we partition into 3 chunks for each regressor,
# which decreases model quality
print(metrics)
#     L1(avg)    L2(avg)  RMS(avg)  Loss-fn(avg)  R Squared
# 0  5.481676  44.924838  6.702599     44.924838   -0.63555
