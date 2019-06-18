###############################################################################
# SsaChangePointDetector
import numpy as np
import pandas as pd
from nimbusml.timeseries import SsaChangePointDetector

# This example creates a time series (list of data with the
# i-th element corresponding to the i-th time slot). 
# The estimator is applied to identify points where data distribution changed.
# This estimator can account for temporal seasonality in the data.

# Generate sample series data with a recurring
# pattern and a spike within the pattern
seasonality_size = 5
seasonal_data = np.arange(seasonality_size)

data = np.tile(seasonal_data, 3)
data = np.append(data, [0, 100, 200, 300, 400]) # change distribution

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
# 16    100
# 17    200
# 18    300
# 19    400

training_seasons = 3
training_size = seasonality_size * training_seasons

cpd = SsaChangePointDetector(confidence=95,
                             change_history_length=8,
                             training_window_size=training_size,
                             seasonal_window_size=seasonality_size + 1) << {'result': 'ts'}

cpd.fit(X_train, verbose=1)
data = cpd.transform(X_train)

print(data)

#      ts  result.Alert  result.Raw Score  result.P-Value Score  result.Martingale Score
# 0     0           0.0         -2.531824          5.000000e-01             1.470334e-06
# 1     1           0.0         -0.008832          5.818072e-03             8.094459e-05
# 2     2           0.0          0.763040          1.374071e-01             2.588526e-04
# 3     3           0.0          0.693811          2.797713e-01             4.365186e-04
# 4     4           0.0          1.442079          1.838294e-01             1.074242e-03
# 5     0           0.0         -1.844414          1.707238e-01             2.825599e-03
# 6     1           0.0          0.219578          4.364025e-01             3.193633e-03
# 7     2           0.0          0.201708          4.505472e-01             3.507451e-03
# 8     3           0.0          0.157089          4.684456e-01             3.719387e-03
# 9     4           0.0          1.329494          1.773046e-01             1.717610e-04
# 10    0           0.0         -1.792391          7.353794e-02             3.014897e-04
# 11    1           0.0          0.161634          4.999295e-01             1.788041e-04
# 12    2           0.0          0.092626          4.953789e-01             7.326680e-05
# 13    3           0.0          0.084648          4.514174e-01             3.053876e-05
# 14    4           0.0          1.305554          1.202619e-01             9.741702e-05
# 15    0           0.0         -1.792391          7.264402e-02             5.034093e-04
# 16  100           1.0         99.161634          1.000000e-08             4.031944e+03   <-- alert is on, predicted change point
# 17  200           0.0        185.229474          5.485437e-04             7.312609e+05
# 18  300           0.0        270.403543          1.259683e-02             3.578470e+06
# 19  400           0.0        357.113747          2.978766e-02             4.529837e+07
