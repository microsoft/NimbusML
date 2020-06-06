###############################################################################
# RobustScaler
import pandas as pd
from nimbusml import Pipeline
from nimbusml.preprocessing.normalization import RobustScaler


df = pd.DataFrame(data=dict(c0=[1, 3, 5, 7, 9]))

xf = RobustScaler(columns='c0', center=True, scale=True)
pipeline = Pipeline([xf])
result = pipeline.fit_transform(df)

print(result)
#     c0
# 0 -1.0
# 1 -0.5
# 2  0.0
# 3  0.5
# 4  1.0
