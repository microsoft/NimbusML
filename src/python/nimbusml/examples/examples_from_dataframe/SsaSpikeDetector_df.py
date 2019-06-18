###############################################################################
# SsaSpikeDetector
import numpy as np
import pandas as pd
from nimbusml.timeseries import SsaSpikeDetector

# This example creates a time series (list of data with the
# i-th element corresponding to the i-th time slot). 
# The estimator is applied to identify spiking points in the series.
# This estimator can account for temporal seasonality in the data.

# Generate sample series data with a recurring
# pattern and a spike within the pattern
seasonality_size = 5
seasonal_data = np.arange(seasonality_size)

data = np.tile(seasonal_data, 3)
data = np.append(data, [100]) # add a spike
data = np.append(data, seasonal_data)

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
# 15    100
# 16      0
# 17      1
# 18      2
# 19      3
# 20      4

training_seasons = 3
training_size = seasonality_size * training_seasons

ssd = SsaSpikeDetector(confidence=95,
                       pvalue_history_length=8,
                       training_window_size=training_size,
                       seasonal_window_size=seasonality_size + 1) << {'result': 'ts'}

ssd.fit(X_train, verbose=1)
data = ssd.transform(X_train)

print(data)

#      ts  result.Alert  result.Raw Score  result.P-Value Score
# 0     0           0.0         -2.531824          5.000000e-01
# 1     1           0.0         -0.008832          5.818072e-03
# 2     2           0.0          0.763040          1.374071e-01
# 3     3           0.0          0.693811          2.797713e-01
# 4     4           0.0          1.442079          1.838294e-01
# 5     0           0.0         -1.844414          1.707238e-01
# 6     1           0.0          0.219578          4.364025e-01
# 7     2           0.0          0.201708          4.505472e-01
# 8     3           0.0          0.157089          4.684456e-01
# 9     4           0.0          1.329494          1.773046e-01
# 10    0           0.0         -1.792391          7.353794e-02
# 11    1           0.0          0.161634          4.999295e-01
# 12    2           0.0          0.092626          4.953789e-01
# 13    3           0.0          0.084648          4.514174e-01
# 14    4           0.0          1.305554          1.202619e-01
# 15  100           1.0         98.207609          1.000000e-08   <-- alert is on, predicted spike
# 16    0           0.0        -13.831450          2.912225e-01
# 17    1           0.0         -1.741884          4.379857e-01
# 18    2           0.0         -0.465426          4.557261e-01
# 19    3           0.0        -16.497133          2.926521e-01
# 20    4           0.0        -29.817375          2.060473e-01
