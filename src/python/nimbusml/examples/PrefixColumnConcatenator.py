###############################################################################
# PrefixColumnConcatenator
import numpy as np
import pandas as pd
from nimbusml.preprocessing.schema import PrefixColumnConcatenator

data = pd.DataFrame(
    data=dict(
        PrefixA=[2.5, np.nan, 2.1, 1.0],
        PrefixB=[.75, .9, .8, .76],
        AnotherColumn=[np.nan, 2.5, 2.6, 2.4]))

# transform usage
xf = PrefixColumnConcatenator(columns={'combined': 'Prefix'})

# fit and transform
features = xf.fit_transform(data)

# print features
print(features.head())
#   PrefixA  PrefixB  AnotherColumn  combined.PrefixA  combined.PrefixB
#0      2.5     0.75            NaN               2.5              0.75
#1      NaN     0.90            2.5               NaN              0.90
#2      2.1     0.80            2.6               2.1              0.80
#3      1.0     0.76            2.4               1.0              0.76