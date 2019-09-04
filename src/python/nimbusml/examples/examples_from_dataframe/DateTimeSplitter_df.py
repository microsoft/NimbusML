###############################################################################
# DateTimeSplitter
import pandas
from nimbusml.preprocessing import DateTimeSplitter

df = pandas.DataFrame(data=dict(
    tokens1=[1, 2, 3, 4],
    tokens2=[10, 11, 12, 13]
))

# duplicate a column
cd = DateTimeSplitter(prefix='dt') << {'tokens3': 'tokens1'}
y = cd.fit_transform(df)

# view the three columns
pandas.set_option('display.max_columns', None)
print(y)
