###############################################################################
# IidSpikeDetector
from nimbusml import Pipeline, FileDataStream
from nimbusml.datasets import get_dataset
from nimbusml.timeseries import IidSpikeDetector

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
    IidSpikeDetector(columns={'t2_spikes': 't2'}, pvalue_history_length=5)
])

result = pipeline.fit_transform(data)
print(result)
#      t1     t2       t3  t2_spikes.Alert  t2_spikes.Raw Score  t2_spikes.P-Value Score
# 0  0.01   0.01   0.0100              0.0                 0.01             5.000000e-01
# 1  0.02   0.02   0.0200              0.0                 0.02             4.960106e-01
# 2  0.03   0.03   0.0200              0.0                 0.03             1.139087e-02
# 3  0.03   0.03   0.0250              0.0                 0.03             2.058296e-01
# 4  0.03   0.03   0.0005              0.0                 0.03             2.804577e-01
# 5  0.03   0.05   0.0100              1.0                 0.05             3.743552e-03
# 6  0.05   0.07   0.0500              1.0                 0.07             4.136079e-03
# 7  0.07   0.09   0.0900              0.0                 0.09             2.242496e-02
# 8  0.09  99.00  99.0000              1.0                99.00             1.000000e-08
# 9  1.10   0.10   0.1000              0.0                 0.10             4.015681e-01

