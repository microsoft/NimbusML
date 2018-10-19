###############################################################################
# SkipFilter: skip first N rows in a dataset
# TakeFilter: take first N rows in a dataset
import pandas
from nimbusml import Pipeline
from nimbusml.preprocessing.filter import SkipFilter, TakeFilter

df = pandas.DataFrame(data=dict(
    review=['row' + str(i) for i in range(10)]))

# skip the first 5 rows
print(SkipFilter(count=5).fit_transform(df))

# take the first 5 rows
print(TakeFilter(count=5).fit_transform(df))

# skip 3 then take 5 rows
pipe = Pipeline([SkipFilter(count=3), TakeFilter(count=5)])
print(pipe.fit_transform(df))
