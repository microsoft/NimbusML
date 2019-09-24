###############################################################################
# DateTimeSplitter
import pandas
from nimbusml.preprocessing import DateTimeSplitter

df = pandas.DataFrame(data=dict(
    tokens1=[1, 2, 3, 157161600],
    tokens2=[10, 11, 12, 13]
))

cols_to_drop = [
    'Hour12', 'DayOfWeek', 'DayOfQuarter',
    'DayOfYear', 'WeekOfMonth', 'QuarterOfYear',
    'HalfOfYear', 'WeekIso', 'YearIso', 'MonthLabel',
    'AmPmLabel', 'DayOfWeekLabel', 'IsPaidTimeOff'
]

cd = DateTimeSplitter(prefix='dt',
                      country='Canada',
                      columns_to_drop=cols_to_drop) << 'tokens1'
y = cd.fit_transform(df)

# view the three columns
pandas.set_option('display.max_columns', None)
pandas.set_option('display.width', 1000)
print(y)
#      tokens1  tokens2  dtYear  dtMonth  dtDay  dtHour  dtMinute  dtSecond  dtAmPm   dtHolidayName
# 0          1       10    1970        1      1       0         0         1       0  New Year's Day
# 1          2       11    1970        1      1       0         0         2       0  New Year's Day
# 2          3       12    1970        1      1       0         0         3       0  New Year's Day
# 3  157161600       13    1974       12     25       0         0         0       0   Christmas Day
