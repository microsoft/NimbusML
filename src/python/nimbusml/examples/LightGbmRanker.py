###############################################################################
# LightGbmRanker
import numpy as np
from nimbusml import Pipeline, FileDataStream
from nimbusml.datasets import get_dataset
from nimbusml.ensemble import LightGbmRanker

# data input (as a FileDataStream)
path = get_dataset('gen_tickettrain').as_filepath()

# LightGbmRanker requires key type for group column
data = FileDataStream.read_csv(path, dtype={'group': np.uint32})

# define the training pipeline
pipeline = Pipeline([LightGbmRanker(
    feature=['Class', 'dep_day', 'duration'], label='rank', group_id='group')])

# train, predict, and evaluate.
# TODO: Replace with CV
metrics, predictions = pipeline \
    .fit(data) \
    .test(data, output_scores=True)

# print predictions
print(predictions.head())
#       Score
# 0 -0.124121
# 1 -0.124121
# 2 -0.124121
# 3 -0.376062
# 4 -0.376062
# print evaluation metrics
print(metrics)
#       NDCG@1     NDCG@2     NDCG@3     DCG@1      DCG@2      DCG@3
# 0  55.238095  65.967598  67.726087  6.492128  11.043324  13.928714
