###############################################################################
# MinMaxScaler
import pandas as pd
from nimbusml.preprocessing.normalization import MinMaxScaler

in_df = pd.DataFrame(
    data=dict(
        Sepal_Length=[
            2.5, 1, 2.1, 1.0], Sepal_Width=[
            .75, .9, .8, .76], Petal_Length=[
            0, 2.5, 2.6, 2.4], Species=[
            "setosa", "viginica", "setosa", 'versicolor']))

# generate two new Columns - Petal_Normed and Sepal_Normed
normed = MinMaxScaler() << {
    'Petal_Normed': 'Petal_Length',
    'Sepal_Normed': 'Sepal_Width'}
out_df = normed.fit_transform(in_df)

print('MinMaxScaler\n', (out_df))
