###############################################################################
# GlobalContrastRowScaler
import numpy as np
import pandas as pd
from nimbusml import Pipeline
from nimbusml.preprocessing.normalization import GlobalContrastRowScaler
from nimbusml.preprocessing.schema import ColumnConcatenator

in_df = pd.DataFrame(
    data=dict(
        Sepal_Length=[
            2.5, 1, 2.1, 1.0], Sepal_Width=[
            .75, .9, .8, .76], Petal_Length=[
            0, 2.5, 2.6, 2.4], Species=[
            "setosa", "viginica", "setosa", 'versicolor']))

in_df.iloc[:, 0:3] = in_df.iloc[:, 0:3].astype(np.float32)

# generate two new Columns - Petal_Normed and Sepal_Normed
concat = ColumnConcatenator() << {
    'concated_columns': [
        'Petal_Length',
        'Sepal_Width',
        'Sepal_Length']}

# performs a global contrast normalization on input values:
# Y = (s * X - M) / D, where s is a scale, M is mean and D is either
# L2 norm or standard deviation.
normed = GlobalContrastRowScaler() << {'normed_columns': 'concated_columns'}

pipeline = Pipeline([concat, normed])
out_df = pipeline.fit_transform(in_df)

print('GlobalContrastRowScaler\n', out_df)
