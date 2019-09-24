###############################################################################
# DateTimeSplitter
import pandas as pd
from nimbusml import FileDataStream
from nimbusml.datasets import get_dataset
from nimbusml.preprocessing import DateTimeSplitter

# data input (as a FileDataStream)
path = get_dataset('infert').as_filepath()

data = FileDataStream.read_csv(path, sep=',')

# transform usage
xf = DateTimeSplitter(prefix='dt_') << 'age'

# fit and transform
features = xf.fit_transform(data)

features = features.drop(['row_num', 'education', 'parity', 'induced',
                          'case', 'spontaneous', 'stratum', 'pooled.stratum'], axis=1)

# print features
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(features.head())
#    age  dt_Year  dt_Month  dt_Day  dt_Hour  dt_Minute  dt_Second  dt_AmPm  dt_Hour12  dt_DayOfWeek  dt_DayOfQuarter  dt_DayOfYear  dt_WeekOfMonth  dt_QuarterOfYear  dt_HalfOfYear  dt_WeekIso  dt_YearIso dt_MonthLabel dt_AmPmLabel dt_DayOfWeekLabel dt_HolidayName  dt_IsPaidTimeOff
# 0   26     1970         1       1        0          0         26        0          0             4                1             0               0                 1              1           1        1970       January           am          Thursday           None                 0
# 1   42     1970         1       1        0          0         42        0          0             4                1             0               0                 1              1           1        1970       January           am          Thursday           None                 0
# 2   39     1970         1       1        0          0         39        0          0             4                1             0               0                 1              1           1        1970       January           am          Thursday           None                 0
# 3   34     1970         1       1        0          0         34        0          0             4                1             0               0                 1              1           1        1970       January           am          Thursday           None                 0
# 4   35     1970         1       1        0          0         35        0          0             4                1             0               0                 1              1           1        1970       January           am          Thursday           None                 0