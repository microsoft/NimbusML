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
    SsaForecaster(series_length=6,
                  train_size=8,
                  window_size=3,
                  horizon=2,
                  columns={'t2_fc': 't2'})
])

result = pipeline.fit_transform(data)

pd.set_option('display.float_format', lambda x: '%.2f' % x)
print(result)

# Output
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

