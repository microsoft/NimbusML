###############################################################################
# DateTimeSplitter
import pandas
from nimbusml.preprocessing import DateTimeSplitter

df = pandas.DataFrame(data=dict(
    tokens1=['one_' + str(i) for i in range(8)],
    tokens2=['two_' + str(i) for i in range(8)]
))

# duplicate a column
cd = DateTimeSplitter(prefix='dt') << {'tokens3': 'tokens1'}
y = cd.fit_transform(df)

# view the three columns
print(y)
