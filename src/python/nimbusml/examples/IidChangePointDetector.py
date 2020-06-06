###############################################################################
# IidChangePointDetector
from nimbusml import Pipeline, FileDataStream
from nimbusml.datasets import get_dataset
from nimbusml.timeseries import IidChangePointDetector

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
    IidChangePointDetector(columns={'t2_cp': 't2'}, change_history_length=4)
])

result = pipeline.fit_transform(data)
print(result)

#      t1     t2       t3  t2_cp.Alert  t2_cp.Raw Score  t2_cp.P-Value Score  t2_cp.Martingale Score
# 0  0.01   0.01   0.0100          0.0             0.01         5.000000e-01            1.212573e-03
# 1  0.02   0.02   0.0200          0.0             0.02         4.960106e-01            1.221347e-03
# 2  0.03   0.03   0.0200          0.0             0.03         1.139087e-02            3.672914e-02
# 3  0.03   0.03   0.0250          0.0             0.03         2.058296e-01            8.164447e-02
# 4  0.03   0.03   0.0005          0.0             0.03         2.804577e-01            1.373786e-01
# 5  0.03   0.05   0.0100          1.0             0.05         1.448886e-06            1.315014e+04
# 6  0.05   0.07   0.0500          0.0             0.07         2.616611e-03            4.941587e+04
# 7  0.07   0.09   0.0900          0.0             0.09         3.053187e-02            2.752614e+05
# 8  0.09  99.00  99.0000          0.0            99.00         1.000000e-08            1.389396e+12
# 9  1.10   0.10   0.1000          1.0             0.10         3.778296e-01            1.854344e+07

