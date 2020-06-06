###############################################################################
# LpScaler
import numpy as np
import pandas as pd
from nimbusml import Pipeline
from nimbusml.preprocessing.normalization import LpScaler
from nimbusml.preprocessing.schema import ColumnConcatenator

in_df = pd.DataFrame(
    data=dict(
        Sepal_Length=[2.5, 1, 2.1, 1.0],
        Sepal_Width=[.75, .9, .8, .76],
        Petal_Length=[0, 2.5, 2.6, 2.4],
        Species=["setosa", "viginica", "setosa", 'versicolor']))

in_df.iloc[:, 0:3] = in_df.iloc[:, 0:3].astype(np.float32)

concat = ColumnConcatenator() << {
    'cat': [ 'Sepal_Length', 'Sepal_Width', 'Petal_Length']
}

# Normalize the input values by rescaling them to unit norm (L2, L1 or LInf).
# Performs the following operation on a vector X: Y = (X - M) / D, where M is
# mean and D is either L2 norm, L1 norm or LInf norm.
normed = LpScaler() << {'norm': 'cat'}

pipeline = Pipeline([concat, normed])
out_df = pipeline.fit_transform(in_df)

print_opts = {
    'index': False,
    'justify': 'left',
    'columns': [
        'Sepal_Length',
        'Sepal_Width',
        'Petal_Length',
        'norm.Sepal_Length',
        'norm.Sepal_Width',
        'norm.Petal_Length'
    ]
}
print('LpScaler\n', out_df.to_string(**print_opts))

# Sepal_Length  Sepal_Width  Petal_Length  norm.Sepal_Length  norm.Sepal_Width  norm.Petal_Length
# 2.5           0.75         0.0           0.957826           0.287348          0.000000
# 1.0           0.90         2.5           0.352235           0.317011          0.880587
# 2.1           0.80         2.6           0.611075           0.232790          0.756569
# 1.0           0.76         2.4           0.369167           0.280567          0.886001
