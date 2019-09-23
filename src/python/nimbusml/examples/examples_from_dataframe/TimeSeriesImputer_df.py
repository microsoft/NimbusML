###############################################################################
# DateTimeSplitter
import pandas
from nimbusml.timeseries import TimeSeriesImputer

df = pandas.DataFrame(data=dict(
    ts=[1, 2, 3, 5],
    grain=[1970, 1970, 1970, 1970],
    c3=[10, 13, 15, 20],
    c4=[19, 12, 16, 19]
))

print(df)

tsi = TimeSeriesImputer(time_series_column='ts',
                        grain_columns=['grain'],
                        filter_columns=['c3', 'c4'],
                        impute_mode='ForwardFill',
                        filter_mode='Include')
result = tsi.fit_transform(df)

print(result)
#    ts  grain  c3  c4  IsRowImputed
# 0   0      0   0   0         False
# 1   1   1970  10  19         False
# 2   2   1970  13  12         False
# 3   3   1970  15  16         False
# 4   4   1970  15  16          True  <== New row added
# 5   5   1970  20  19         False
