###############################################################################
# Filter
import numpy as np
import pandas as pd
from nimbusml import FileDataStream
from nimbusml.preprocessing.missing_values import Filter

with_nans = pd.DataFrame(
    data=dict(
        Sepal_Length=[
            2.5, np.nan, 2.1, 1.0], Sepal_Width=[
            .75, .9, .8, .76], Petal_Length=[
            np.nan, 2.5, 2.6, 2.4], Petal_Width=[
            .8, .7, .9, 0.7], Species=[
            "setosa", "viginica", "", 'versicolor']))

# write NaNs to file to see how transforms work
tmpfile = 'tmpfile_with_nans.csv'
with_nans.to_csv(tmpfile, index=False)

# schema for reading directly from text files
schema = 'sep=, col=Petal_Length:R4:0 col=Petal_Width:R4:1' \
         'col=Sepal_Length:R4:2 col=Sepal_Width:R4:3 col=Species:TX:4 header+'
data = FileDataStream.read_csv(tmpfile)
print(data.schema)

# filter out rows where Sepal_Length is NaN
nafilter = Filter() << ['Sepal_Length', 'Petal_Length']

print(with_nans)
print('NAFilter\n', nafilter.fit_transform(data))
