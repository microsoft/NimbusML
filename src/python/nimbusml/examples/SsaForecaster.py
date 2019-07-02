###############################################################################
# SsaForecaster
import pandas as pd
from nimbusml import Pipeline, FileDataStream
from nimbusml.datasets import get_dataset
from nimbusml.timeseries import SsaForecaster

# data input (as a FileDataStream)
path = get_dataset('timeseries').as_filepath()

data = FileDataStream.read_csv(path)
print(data.head())
#      t1    t2      t3
# 0  0.01  0.01  0.0100
# 1  0.02  0.02  0.0200
# 2  0.03  0.03  0.0200
# 3  0.03  0.03  0.0250
# 4  0.03  0.03  0.0005

# define the training pipeline
pipeline = Pipeline([
<<<<<<< HEAD
    SsaForecaster(series_length=6,
                  train_size=8,
                  window_size=3,
                  horizon=2,
=======
    SsaForecaster(forcasting_confident_lower_bound_column_name="cmin",
                  forcasting_confident_upper_bound_column_name="cmax",
                  series_length=6,
                  train_size=8,
                  window_size=3,
                  horizon=2,
                  # max_growth={'TimeSpan': 1, 'Growth': 1}
>>>>>>> Fixed estimator checks failed from ssa_forecasting's update (#166)
                  columns={'t2_fc': 't2'})
])

result = pipeline.fit_transform(data)

pd.set_option('display.float_format', lambda x: '%.2f' % x)
print(result)

# Output
<<<<<<< HEAD
#
#     t1    t2    t3  t2_fc.0  t2_fc.1
# 0 0.01  0.01  0.01     0.10     0.12
# 1 0.02  0.02  0.02     0.06     0.08
# 2 0.03  0.03  0.02     0.04     0.05
# 3 0.03  0.03  0.03     0.05     0.06
# 4 0.03  0.03  0.00     0.05     0.07
# 5 0.03  0.05  0.01     0.06     0.08
# 6 0.05  0.07  0.05     0.09     0.12
# 7 0.07  0.09  0.09     0.12     0.16
# 8 0.09 99.00 99.00    57.92    82.88
# 9 1.10  0.10  0.10    60.50    77.18
=======
#     t1    t2    t3  t2_fc.0  t2_fc.1  cmin.0  cmin.1  cmax.0  cmax.1
# 0 0.01  0.01  0.01     0.10     0.12    0.09    0.11    0.11    0.13
# 1 0.02  0.02  0.02     0.06     0.08    0.06    0.07    0.07    0.09
# 2 0.03  0.03  0.02     0.04     0.05    0.03    0.04    0.05    0.07
# 3 0.03  0.03  0.03     0.05     0.06    0.04    0.05    0.05    0.07
# 4 0.03  0.03  0.00     0.05     0.07    0.04    0.05    0.06    0.08
# 5 0.03  0.05  0.01     0.06     0.08    0.05    0.07    0.07    0.10
# 6 0.05  0.07  0.05     0.09     0.12    0.08    0.10    0.10    0.13
# 7 0.07  0.09  0.09     0.12     0.16    0.11    0.15    0.13    0.17
# 8 0.09 99.00 99.00    57.92    82.88   57.91   82.87   57.93   82.89
# 9 1.10  0.10  0.10    60.50    77.18   60.49   77.17   60.50   77.19
>>>>>>> Fixed estimator checks failed from ssa_forecasting's update (#166)

