###############################################################################
# SsaForecaster
import numpy as np
import pandas as pd
from nimbusml.timeseries import SsaForecaster

# This example creates a time series (list of data with the
# i-th element corresponding to the i-th time slot).

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

x_test = X_train.copy()
x_test[-3:] = [100, 110, 120]

# x_test looks like this
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
# 12    100
# 13    110
# 14    120

training_seasons = 3
training_size = seasonality_size * training_seasons

forecaster = SsaForecaster(series_length=8,
                           train_size=training_size,
                           window_size=seasonality_size + 1,
                           horizon=4) <<   {'fc': 'ts'}

forecaster.fit(X_train, verbose=1)
data = forecaster.transform(x_test)

pd.set_option('display.float_format', lambda x: '%.2f' % x)
print(data)

# The fc.x columns are the forecasts
# given the input in the ts column.
#
#      ts  fc.0  fc.1  fc.2  fc.3
# 0     0  1.00  2.00  3.00  4.00
# 1     1  2.00  3.00  4.00 -0.00
# 2     2  3.00  4.00 -0.00  1.00
# 3     3  4.00 -0.00  1.00  2.00
# 4     4 -0.00  1.00  2.00  3.00
# 5     0  1.00  2.00  3.00  4.00
# 6     1  2.00  3.00  4.00 -0.00
# 7     2  3.00  4.00 -0.00  1.00
# 8     3  4.00 -0.00  1.00  2.00
# 9     4 -0.00  1.00  2.00  3.00
# 10    0  1.00  2.00  3.00  4.00
# 11    1  2.00  3.00  4.00 -0.00
# 12  100  3.00  4.00  0.00  1.00
# 13  110  4.00 -0.00  1.00 75.50
# 14  120 -0.00  1.00 83.67 83.25
