###############################################################################
# DateTimeSplitter
import pandas
from nimbusml.preprocessing import DateTimeSplitter

df = pandas.DataFrame(data=dict(
    tokens1=[1, 2, 3, 4],
    tokens2=[10, 11, 12, 13]
))

# duplicate a column
cd = DateTimeSplitter(prefix='dt') << 'tokens1'
y = cd.fit_transform(df)

# view the three columns
pandas.set_option('display.max_columns', None)
pandas.set_option('display.width', 1000)
print(y)
#    tokens1  tokens2  dtYear  dtMonth  dtDay  dtHour  dtMinute  dtSecond  dtAmPm  dtHour12  dtDayOfWeek  dtDayOfQuarter  dtDayOfYear  dtWeekOfMonth  dtQuarterOfYear  dtHalfOfYear  dtWeekIso  dtYearIso dtMonthLabel dtAmPmLabel dtDayOfWeekLabel    dtHolidayName  dtIsPaidTimeOff
# 0        1       10    1970        1      1       0         0         1       0         0            4               1            0              0                1             1          1       1970      January          am         Thursday  NOT IMPLEMENTED                0
# 1        2       11    1970        1      1       0         0         2       0         0            4               1            0              0                1             1          1       1970      January          am         Thursday  NOT IMPLEMENTED                0
# 2        3       12    1970        1      1       0         0         3       0         0            4               1            0              0                1             1          1       1970      January          am         Thursday  NOT IMPLEMENTED                0
# 3        4       13    1970        1      1       0         0         4       0         0            4               1            0              0                1             1          1       1970      January          am         Thursday  NOT IMPLEMENTED                0
