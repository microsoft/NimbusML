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
