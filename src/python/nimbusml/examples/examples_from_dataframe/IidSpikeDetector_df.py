###############################################################################
# IidSpikeDetector
import numpy as np
import pandas as pd
from nimbusml.time_series import IidSpikeDetector

X_train = pd.Series([5,5,5,5,5,10,5,5,5,5,5])

isd = IidSpikeDetector()
result = isd.fit_transform(X_train)

print(result)
