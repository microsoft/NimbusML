###############################################################################
# IidChangePointDetector
import pandas as pd
from nimbusml.timeseries import IidChangePointDetector

# Create a sample series with a change
input_data = [5, 5, 5, 5, 5, 5, 5, 5]
input_data.extend([7, 7, 7, 7, 7, 7, 7, 7])

X_train = pd.Series(input_data, name="ts")

cpd = IidChangePointDetector(confidence=95, change_history_length=4) << {'result': 'ts'}
data = cpd.fit_transform(X_train)

print(data)

#     ts  result.Alert  result.Raw Score  result.P-Value Score  result.Martingale Score
# 0    5           0.0               5.0          5.000000e-01                 0.001213
# 1    5           0.0               5.0          5.000000e-01                 0.001213
# 2    5           0.0               5.0          5.000000e-01                 0.001213
# 3    5           0.0               5.0          5.000000e-01                 0.001213
# 4    5           0.0               5.0          5.000000e-01                 0.001213
# 5    5           0.0               5.0          5.000000e-01                 0.001213
# 6    5           0.0               5.0          5.000000e-01                 0.001213
# 7    5           0.0               5.0          5.000000e-01                 0.001213
# 8    7           1.0               7.0          1.000000e-08             10298.666376   <-- alert is on, predicted changepoint
# 9    7           0.0               7.0          1.328455e-01             33950.164799
# 10   7           0.0               7.0          2.613750e-01             60866.342063
# 11   7           0.0               7.0          3.776152e-01             78362.038772
# 12   7           0.0               7.0          5.000000e-01                 0.009226
# 13   7           0.0               7.0          5.000000e-01                 0.002799
# 14   7           0.0               7.0          5.000000e-01                 0.001561
# 15   7           0.0               7.0          5.000000e-01                 0.001213

