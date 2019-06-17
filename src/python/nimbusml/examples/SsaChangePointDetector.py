###############################################################################
# SsaChangePointDetector
from nimbusml import Pipeline, FileDataStream
from nimbusml.datasets import get_dataset
from nimbusml.time_series import SsaChangePointDetector

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
    SsaChangePointDetector(columns={'t2_cp': 't2'},
                           change_history_length=4,
                           training_window_size=8,
                           seasonal_window_size=3)
])

result = pipeline.fit_transform(data)
print(result)

#      t1     t2       t3  t2_cp.Alert  t2_cp.Raw Score  t2_cp.P-Value Score  t2_cp.Martingale Score
# 0  0.01   0.01   0.0100          0.0        -0.111334         5.000000e-01                0.001213
# 1  0.02   0.02   0.0200          0.0        -0.076755         4.862075e-01                0.001243
# 2  0.03   0.03   0.0200          0.0        -0.034871         3.856320e-03                0.099119
# 3  0.03   0.03   0.0250          0.0        -0.012559         8.617091e-02                0.482400
# 4  0.03   0.03   0.0005          0.0        -0.015723         2.252377e-01                0.988788
# 5  0.03   0.05   0.0100          0.0        -0.001133         1.767711e-01                2.457946
# 6  0.05   0.07   0.0500          0.0         0.006265         9.170460e-02                0.141898
# 7  0.07   0.09   0.0900          0.0         0.002383         2.701134e-01                0.050747
# 8  0.09  99.00  99.0000          1.0        98.879520         1.000000e-08           210274.372059
# 9  1.10   0.10   0.1000          0.0       -57.817568         6.635692e-02           507877.454862
