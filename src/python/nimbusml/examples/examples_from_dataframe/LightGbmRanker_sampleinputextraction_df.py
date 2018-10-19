###############################################################################
# LightGbmRanker
import numpy as np
import pandas as pd
from nimbusml import Pipeline, Role
from nimbusml.datasets import get_dataset
from nimbusml.ensemble import LightGbmRanker

np.random.seed(0)
file_path = get_dataset("gen_tickettrain").as_filepath()

df = pd.read_csv(file_path)
df['group'] = df['group'].astype(np.uint32)

X = df.drop(['rank'], axis=1)
y = df['rank']

e = Pipeline([LightGbmRanker() << {Role.Feature: [
    'Class', 'dep_day', 'duration'], Role.Label: 'rank',
    Role.GroupId: 'group'}])

e.fit(df)

# test
metrics, scores = e.test(X, y, evaltype='ranking',
                         group_id='group', output_scores=True)
print(metrics)
