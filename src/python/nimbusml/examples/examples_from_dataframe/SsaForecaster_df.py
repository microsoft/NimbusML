###############################################################################
# SsaForecaster
import numpy as np
import pandas as pd
from nimbusml.timeseries import SsaForecaster

# This example creates a time series (list of data with the
# i-th element corresponding to the i-th time slot). 
# The estimator is applied to identify points where data distribution changed.
# This estimator can account for temporal seasonality in the data.

# Generate sample series data with a recurring pattern
seasonality_size = 5
seasonal_data = np.arange(seasonality_size)

data = np.tile(seasonal_data, 3)
X_train = pd.Series(data, name="ts")

# X_train looks like this
# 0       0
# 1       1
# 2       2
# 3       3
# 4       4
# 5       0
# 6       1
# 7       2
# 8       3
# 9       4
# 10      0
# 11      1
# 12      2
# 13      3
# 14      4
# 15      0

training_seasons = 3
training_size = seasonality_size * training_seasons

cpd = SsaForecaster(confidence=95,
                    series_length=8,
                    train_size=training_size,
                    window_size=seasonality_size + 1) << {'result': 'ts'}

cpd.fit(X_train, verbose=1)
data = cpd.transform(X_train)

print(data)
