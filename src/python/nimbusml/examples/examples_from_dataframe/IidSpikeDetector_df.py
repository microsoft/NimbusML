###############################################################################
# IidSpikeDetector
import pandas as pd
from nimbusml.timeseries import IidSpikeDetector

X_train = pd.Series([5, 5, 5, 5, 5, 10, 5, 5, 5, 5, 5], name="ts")

isd = IidSpikeDetector(confidence=95, pvalue_history_length=2.5) << {'result': 'ts'}

isd.fit(X_train, verbose=1)
data = isd.transform(X_train)

print(data)

#       ts  result.Alert  result.Raw Score  result.P-Value Score
# 0    5.0           0.0               5.0          5.000000e-01
# 1    5.0           0.0               5.0          5.000000e-01
# 2    5.0           0.0               5.0          5.000000e-01
# 3    5.0           0.0               5.0          5.000000e-01
# 4    5.0           0.0               5.0          5.000000e-01
# 5   10.0           1.0              10.0          1.000000e-08   <-- alert is on, predicted spike
# 6    5.0           0.0               5.0          2.613750e-01
# 7    5.0           0.0               5.0          2.613750e-01
# 8    5.0           0.0               5.0          5.000000e-01
# 9    5.0           0.0               5.0          5.000000e-01
# 10   5.0           0.0               5.0          5.000000e-01
