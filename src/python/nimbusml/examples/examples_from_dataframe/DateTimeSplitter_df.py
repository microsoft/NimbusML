###############################################################################
# DateTimeSplitter
import pandas
from nimbusml import Pipeline
from nimbusml.preprocessing import DateTimeSplitter
from nimbusml.preprocessing.schema import ColumnSelector

df = pandas.DataFrame(data=dict(
    tokens1=[1, 2, 3, 157161600],
    tokens2=[10, 11, 12, 13]
))

cols_to_drop = [
    'dtHour12', 'dtDayOfWeek', 'dtDayOfQuarter',
    'dtDayOfYear', 'dtWeekOfMonth', 'dtQuarterOfYear',
    'dtHalfOfYear', 'dtWeekIso', 'dtYearIso', 'dtMonthLabel',
    'dtAmPmLabel', 'dtDayOfWeekLabel', 'dtIsPaidTimeOff'
]

dts = DateTimeSplitter(prefix='dt', country='Canada') << 'tokens1'

pipeline = Pipeline([dts, ColumnSelector(drop_columns=cols_to_drop)])
y = pipeline.fit_transform(df)

# view the three columns
pandas.set_option('display.max_columns', None)
pandas.set_option('display.width', 1000)
print(y)
#      tokens1  tokens2  dtYear  dtMonth  dtDay  dtHour  dtMinute  dtSecond  dtAmPm   dtHolidayName
# 0          1       10    1970        1      1       0         0         1       0  New Year's Day
# 1          2       11    1970        1      1       0         0         2       0  New Year's Day
# 2          3       12    1970        1      1       0         0         3       0  New Year's Day
# 3  157161600       13    1974       12     25       0         0         0       0   Christmas Day
