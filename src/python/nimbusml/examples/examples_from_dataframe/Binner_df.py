###############################################################################
# Binner
import pandas as pd
from nimbusml.preprocessing.normalization import Binner

in_df = pd.DataFrame(
    data=dict(
        Sepal_Length=[
            2.5, 1, 2.1, 1.0, 0.8, 1.1], Sepal_Width=[
            .75, .9, .8, .76, .85, 0.76], Petal_Length=[
            0, 2.5, 2.6, 2.4, 2.1, 2.2], Species=[
            "setosa", "viginica", "setosa", "versicolor", "viginica",
            "versicolor"]))

# generate two new Columns - Petal_Normed and Sepal_Normed
# bin into 5 bins  using equal width binning (.0, .25, .50, .75, 1.0)
normed = Binner(
    num_bins=5) << {
             'Petal_Normed': 'Petal_Length',
             'Sepal_Normed': 'Sepal_Width'}
out_df = normed.fit(in_df).transform(in_df)

print('Binner\n', (out_df))
