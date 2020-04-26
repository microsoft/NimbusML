# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import os
import sys
import io
import platform
import tempfile
import unittest

import numpy as np
import pandas as pd
from nimbusml import Pipeline
from nimbusml.preprocessing.schema import ColumnSelector
from nimbusml.preprocessing import ToString, ToKeyImputer, DateTimeSplitter
from scipy.sparse import csr_matrix
from nimbusml.timeseries import TimeSeriesImputer, LagLeadOperator, RollingWindow, ForecastingPivot, ShortDrop
from nimbusml.preprocessing import (TensorFlowScorer, FromKey, ToKey,
                                    DateTimeSplitter, OnnxRunner)
import onnxruntime as rt
from data_frame_tool import DataFrameTool as DFT

TEST_CASES = {
    'DateTimeSplitter_Simple',
    'DateTimeSplitter_Complex',
    'ToKey_SimpleFloat',
    'ToKey_SimpleDouble',
    'ToKey_SimpleString',
    'ToKey_2col_Double',
    'ToKey_2col_Double_String',
    'ToString_Numbers',
    'ToString_Other_Types',
    'TimeSeriesImputer_1grain_2gap',
    'TimeSeriesImputer_1grain_1gap_2filtercolumn',
    'TimeSeriesImputer_2grain_nogap',
    'TimeSeriesImputer_2grain_2gap',
    'RollingWindow_Simple_Mean',
    'RollingWindow_Simple_Min',
    'RollingWindow_Simple_Max',
    'RollingWindow_Multiple_Grain',
    'RollingWindow_NonString_Grain',
    'LagLead_Simple_Lag',
    'LagLead_Simple_Lead',
    'LagLead_Complex',
    'ShortGrainDropper',
    'RollingWin_Pivot_Integration',
    'Laglead_Pivot_Integration',
}

INSTANCES = {
    'DateTimeSplitter_Simple': Pipeline([
        DateTimeSplitter(prefix='dt') << 'tokens1',
        ColumnSelector(drop_columns=[
                                    'dtHour12', 'dtDayOfWeek', 'dtDayOfQuarter',
                                    'dtDayOfYear', 'dtWeekOfMonth', 'dtQuarterOfYear',
                                    'dtHalfOfYear', 'dtWeekIso', 'dtYearIso', 'dtMonthLabel',
                                    'dtAmPmLabel', 'dtDayOfWeekLabel', 'dtIsPaidTimeOff'
        ])
    ]),
    'DateTimeSplitter_Complex' : Pipeline([
        DateTimeSplitter(prefix='dt') << 'tokens1',
        ColumnSelector(drop_columns=[
                                    'dtHour12', 'dtDayOfWeek', 'dtDayOfQuarter',
                                    'dtDayOfYear', 'dtWeekOfMonth', 'dtQuarterOfYear',
                                    'dtHalfOfYear', 'dtWeekIso', 'dtYearIso', 'dtMonthLabel',
                                    'dtAmPmLabel', 'dtDayOfWeekLabel', 'dtIsPaidTimeOff'
        ])
    ]),
    'ToKey_SimpleFloat': ToKeyImputer(),
    'ToKey_SimpleDouble': ToKeyImputer(),
    'ToKey_SimpleString': ToKeyImputer(),
    'ToKey_2col_Double': ToKeyImputer(),
    'ToKey_2col_Double_String': ToKeyImputer(),
    'ToString_Numbers': ToString(columns={'f0.out': 'f0',
                                          'f1.out': 'f1',
                                          'f2.out': 'f2'}),
    'ToString_Other_Types': ToString(columns={'f0.out': 'f0',
                                              'f1.out': 'f1'}),
    'TimeSeriesImputer_1grain_2gap': TimeSeriesImputer(time_series_column='ts',
                                                       filter_columns=['c'],
                                                       grain_columns=['grain'],
                                                       impute_mode='ForwardFill',
                                                       filter_mode='Include'),
    'TimeSeriesImputer_1grain_1gap_2filtercolumn': TimeSeriesImputer(time_series_column='ts',
                                                        grain_columns=['grain'],
                                                        filter_columns=['c3', 'c4'],
                                                        impute_mode='ForwardFill',
                                                        filter_mode='Include'),
    'TimeSeriesImputer_2grain_nogap': TimeSeriesImputer(time_series_column='ts',
                                                       filter_columns=['c'],
                                                       grain_columns=['grain'],
                                                       impute_mode='ForwardFill',
                                                       filter_mode='Include'),
    'TimeSeriesImputer_2grain_2gap': TimeSeriesImputer(time_series_column='ts',
                                                       filter_columns=['c'],
                                                       grain_columns=['grain'],
                                                       impute_mode='ForwardFill',
                                                       filter_mode='Include'),
    'RollingWindow_Simple_Mean': RollingWindow(columns={'colA1': 'colA'},
                                               grain_column=['grainA'], 
                                               window_calculation='Mean',
                                               max_window_size=2,
                                               horizon=2),
    'RollingWindow_Simple_Min': RollingWindow(columns={'colA1': 'colA'},
                                              grain_column=['grainA'], 
                                              window_calculation='Min',
                                              max_window_size=1,
                                              horizon=1),
    'RollingWindow_Simple_Max': RollingWindow(columns={'colA1': 'colA'},
                                              grain_column=['grainA'], 
                                              window_calculation='Max',
                                              max_window_size=1,
                                              horizon=1),
    'RollingWindow_Multiple_Grain': RollingWindow(columns={'colA1': 'colA'},
                                                  grain_column=['grainA'], 
                                                  window_calculation='Mean',
                                                  max_window_size=1,
                                                  horizon=1),
    'RollingWindow_NonString_Grain': RollingWindow(columns={'colA1': 'colA'},
                                                   grain_column=['grainA'], 
                                                   window_calculation='Mean',
                                                   max_window_size=1,
                                                   horizon=1),
    'LagLead_Simple_Lag': LagLeadOperator(columns={'colA1': 'colA'},
                                          grain_columns=['grainA'], 
                                          offsets=[-2, -1],
                                          horizon=2),
    'LagLead_Simple_Lead': LagLeadOperator(columns={'colA1': 'colA'},
                                           grain_columns=['grainA'], 
                                           offsets=[1, 2],
                                           horizon=2),
    'LagLead_Complex': LagLeadOperator(columns={'colA1': 'colA'},
                                       grain_columns=['grainA'], 
                                       offsets=[-2, -1, 1, 2],
                                       horizon=2),
    'ShortGrainDropper': ShortDrop(columns={'colA1': 'colA'},
                                   grain_columns=['grainA'],
                                   min_rows=2),
    'RollingWin_Pivot_Integration': Pipeline([
        RollingWindow(columns={'colA1': 'colA'},
                           grain_column=['grainA'], 
                           window_calculation='Mean',
                           max_window_size=2,
                           horizon=2),
        ForecastingPivot(columns_to_pivot=['colA1'])
    ]),
    'Laglead_Pivot_Integration': Pipeline([
        LagLeadOperator(columns={'colA1': 'colA'},
                        grain_columns=['grainA'], 
                        offsets=[-2, -1],
                        horizon=2),
        ForecastingPivot(columns_to_pivot=['colA1'])
    ]),
}
TRAINING_DATASETS = {
    'DateTimeSplitter_Simple': pd.DataFrame(data=dict(
                                tokens1=[1, 2, 3, 157161600]
                               )),
    'DateTimeSplitter_Complex': pd.DataFrame(data=dict(
                                tokens1=[217081624, 1751241600, 217081625, 32445842582]
                               )),
    'ToKey_SimpleFloat': pd.DataFrame(data=dict(
                                    target=[1.0, 1.0, 1.0, 2.0]
                        )).astype({'target': np.float64}),
    'ToKey_SimpleDouble': pd.DataFrame(data=dict(
                                    target=[1.0, 1.0, 1.0, 2.0]
                        )).astype({'target': np.double}),
    'ToKey_SimpleString': pd.DataFrame(data=dict(
                                    target=["one", "one", "one", "two"]
                        )),
    'ToKey_2col_Double': pd.DataFrame(data=dict(
                                    data1=[1.0, 1.0, 1.0, 2.0],
                                    data2=[2.0, 2.0, 2.0, 3.0],
                        )).astype({'data1': np.double,
                                   'data1': np.double}),
    'ToKey_2col_Double_String': pd.DataFrame(data=dict(
                                    data1=[1.0, 1.0, 1.0, 2.0],
                                    data2=["two", "two", "three", "two"],
                        )),
    'ToString_Numbers': pd.DataFrame(data=dict(
                                    f0= [4, 4, -1, 9],
                                    f1= [5, 5, 3.1, -0.23],
                                    f2= [6, 6.7, np.nan, np.nan]
                        )).astype({'f0': np.int32,
                                   'f1': np.float32,
                                   'f2': np.float64}),
    'ToString_Other_Types': pd.DataFrame(data=dict(
                                    f0= [True, False],
                                    f1= [123.45, 135453984983490.5473]
                        )).astype({'f0': bool,
                                   'f1': np.double}),
    'TimeSeriesImputer_1grain_2gap': pd.DataFrame(data=dict(
                                        ts=[1, 2, 3, 5, 7],
                                        grain=[1970, 1970, 1970, 1970, 1970],
                                        c=[10, 11, 12, 13, 14]
                                    )).astype({'ts': np.int64, 'grain': np.int32, 'c': np.int32}),
    'TimeSeriesImputer_1grain_1gap_2filtercolumn': pd.DataFrame(data=dict(
                                        ts=[1, 2, 3, 5],
                                        grain=[1970, 1970, 1970, 1970],
                                        c3=[10, 13, 15, 20],
                                        c4=[19, 12, 16, 19]
                                    )).astype({'ts': np.int64, 'grain': np.int32, 'c3': np.int32, 'c4': np.int32}),
    'TimeSeriesImputer_2grain_nogap': pd.DataFrame(data=dict(
                                        ts=[1, 5, 2, 6],
                                        grain=[1970, 1971, 1970, 1971],
                                        c=[10, 11, 12, 13]
                                    )).astype({'ts': np.int64, 'grain': np.int32, 'c': np.int32}),
    'TimeSeriesImputer_2grain_2gap': pd.DataFrame(data=dict(
                                         ts=[0, 5, 1, 6],
                                         grain=[1970, 1971, 1970, 1971],
                                         c=[10, 11, 12, 13]
                                     )).astype({'ts': np.int64, 'grain': np.int32, 'c': np.int32}),
    'RollingWindow_Simple_Mean': pd.DataFrame(data=dict(
                                     colA=[1.0, 2.0, 3.0, 4.0],
                                     grainA=["one", "one", "one", "one"]
                                 )),
    'RollingWindow_Simple_Min': pd.DataFrame(data=dict(
                                    colA=[1.0, 2.0, 3.0, 4.0],
                                    grainA=["one", "one", "one", "one"]
                                )),
    'RollingWindow_Simple_Max': pd.DataFrame(data=dict(
                                    colA=[1.0, 2.0, 3.0, 4.0],
                                    grainA=["one", "one", "one", "one"]
                                )),
    'RollingWindow_Multiple_Grain': pd.DataFrame(data=dict(
                                        colA=[1.0, 2.0, 1.0, 2.0],
                                        grainA=["one", "one", "two", "two"]
                                    )),
    'RollingWindow_NonString_Grain': pd.DataFrame(data=dict(
                                         colA=[1.0, 2.0, 3.0, 4.0],
                                         grainA=[True, True, True, True]
                                     )),
    'LagLead_Simple_Lag': pd.DataFrame(data=dict(
                              colA=[1.0, 2.0, 3.0, 4.0],
                              grainA=["one", "one", "one", "one"]
                          )),
    'LagLead_Simple_Lead': pd.DataFrame(data=dict(
                               colA=[1.0, 2.0, 3.0, 4.0],
                               grainA=["one", "one", "one", "one"]
                           )),
    'LagLead_Complex': pd.DataFrame(data=dict(
                           colA=[1.0, 2.0, 3.0, 4.0],
                           grainA=["one", "one", "one", "one"]
                       )),
    'ShortGrainDropper': pd.DataFrame(data=dict(
                                    colA=[1.0, 2.0, 3.0, 4.0],
                                    grainA=["one", "one", "one", "two"]
                                )),
    'RollingWin_Pivot_Integration': pd.DataFrame(data=dict(
                                    colA=[1.0, 2.0, 3.0, 4.0],
                                    grainA=["one", "one", "one", "one"]
                                )),
    'Laglead_Pivot_Integration': pd.DataFrame(data=dict(
                                    colA=[1.0, 2.0, 3.0, 4.0],
                                    grainA=["one", "one", "one", "one"]
                                ))
}

INFERENCING_DATASETS = {
    'DateTimeSplitter_Simple': {"tokens1": np.array([1, 2, 3, 157161600]).astype(np.int64).reshape(4,1)},
    'DateTimeSplitter_Complex': {"tokens1": np.array([217081624, 1751241600, 217081625, 32445842582]).astype(np.int64).reshape(4,1)},
    'ToKey_SimpleFloat': {"target": np.array([1, float("NaN"), 4, 7, float("NaN")]).astype(np.float64).reshape(5,1)},
    'ToKey_SimpleDouble': {"target": np.array([1, float("NaN"), 4, 7, float("NaN")]).astype(np.double).reshape(5,1)},
    'ToKey_SimpleString': {"target": np.array(["1", "", "Hello", "", "World"]).reshape(5,1)},
    'ToKey_2col_Double': {"data1": np.array([1.0, float("NaN"), 4.0, 7.0, float("NaN")]).astype(np.double).reshape(5,1),
                          "data2": np.array([1.0, float("NaN"), 4.0, 7.0, float("NaN")]).astype(np.double).reshape(5,1)},
    'ToKey_2col_Double_String': {"data1": np.array([1.0, float("NaN"), 4.0, 7.0, float("NaN")]).astype(np.double).reshape(5,1),
                                 "data2": np.array(["1", "", "Hello", "", "World"]).reshape(5,1)},
    'ToString_Numbers': {"f0": np.array([4, 4, -1, 9]).astype(np.int32).reshape(4,1),
                         "f1": np.array([5, 5, 3.1, -0.23]).astype(np.float32).reshape(4,1),
                         "f2": np.array([6, 6.7, float("NaN"), float("NaN")]).astype(np.float64).reshape(4,1)},
    'ToString_Other_Types': {"f0": np.array([True, False]).astype(bool).reshape(2,1), "f1": np.array([123.45, 135453984983490.5473]).astype(np.double).reshape(2,1)},
    'TimeSeriesImputer_1grain_2gap': {"ts": np.array([1, 2, 3, 5, 7]).astype(np.int64).reshape(5,1), "grain": np.array([1970, 1970, 1970, 1970, 1970]).astype(np.int32).reshape(5,1), 'c': np.array([10, 11, 12, 13, 14]).astype(np.int32).reshape(5,1)},
    'TimeSeriesImputer_1grain_1gap_2filtercolumn': {"ts": np.array([1, 2, 3, 5]).astype(np.int64).reshape(4,1), "grain": np.array([1970, 1970, 1970, 1970]).astype(np.int32).reshape(4,1),
                                                    "c3": np.array([10, 13, 15, 20]).astype(np.int32).reshape(4,1), "c4": np.array([19, 12, 16, 19]).astype(np.int32).reshape(4,1)},
    'TimeSeriesImputer_2grain_nogap': {"ts": np.array([1, 5, 2, 6]).astype(np.int64).reshape(4,1), "grain": np.array([1970, 1971, 1970, 1971]).astype(np.int32).reshape(4,1), 'c': np.array([10, 11, 12, 13]).astype(np.int32).reshape(4,1)},
    'TimeSeriesImputer_2grain_2gap': {"ts": np.array([0, 5, 3, 8]).astype(np.int64).reshape(4,1), "grain": np.array([1970, 1971, 1970, 1971]).astype(np.int32).reshape(4,1), 'c': np.array([10, 11, 12, 13]).astype(np.int32).reshape(4,1)},
    'RollingWindow_Simple_Mean': {"grainA": np.array(["one", "one", "one", "one"]).reshape(4,1), "colA": np.array([1.0, 2.0, 3.0, 4.0]).astype(np.double).reshape(4,1)},
    'RollingWindow_Simple_Min': {"grainA": np.array(["one", "one", "one", "one"]).reshape(4,1), "colA": np.array([1.0, 2.0, 3.0, 4.0]).astype(np.double).reshape(4,1)},
    'RollingWindow_Simple_Max': {"grainA": np.array(["one", "one", "one", "one"]).reshape(4,1), "colA": np.array([1.0, 2.0, 3.0, 4.0]).astype(np.double).reshape(4,1)},
    'RollingWindow_Multiple_Grain': {"grainA": np.array(["one", "one", "two", "two"]).reshape(4,1), "colA": np.array([1.0, 2.0, 1.0, 2.0]).astype(np.double).reshape(4,1)},
    'RollingWindow_NonString_Grain': {"grainA": np.array([True, True, True, True]).reshape(4,1), "colA": np.array([1.0, 2.0, 3.0, 4.0]).astype(np.double).reshape(4,1)},
    'LagLead_Simple_Lag': {"grainA": np.array(["one", "one", "one", "one"]).reshape(4,1), "colA": np.array([1, 2, 3, 4]).astype(np.double).reshape(4,1)},
    'LagLead_Simple_Lead': {"grainA": np.array(["one", "one", "one", "one"]).reshape(4,1), "colA": np.array([1, 2, 3, 4]).astype(np.double).reshape(4,1)},
    'LagLead_Complex': {"grainA": np.array(["one", "one", "one", "one"]).reshape(4,1), "colA": np.array([1, 2, 3, 4]).astype(np.double).reshape(4,1)},
    'ShortGrainDropper': {"grainA": np.array(["one", "one", "one", "two"]).reshape(4,1), "colA": np.array([1, 2, 3, 4]).astype(np.double).reshape(4,1)},
    'RollingWin_Pivot_Integration': {"colA": np.array([1.0, 3.0, 5.0, 7.0]).reshape(4,1), "grainA": np.array(['1970', '1970', '1970', '1970']).reshape(4,1)},
    'Laglead_Pivot_Integration': {"colA": np.array([1, 2, 3, 4]).astype(np.double).reshape(4,1), "grainA": np.array(["one", "one", "one", "one"]).reshape(4,1)},
}
EXPECTED_RESULT = {
    'DateTimeSplitter_Simple': [np.array([1970, 1970, 1970, 1974]).reshape(4, 1), np.array([1, 1, 1, 12]).reshape(4, 1),
                                np.array([1, 1, 1, 25]).reshape(4, 1), np.array([0, 0, 0, 0]).reshape(4, 1),
                                np.array([0, 0, 0, 0]).reshape(4, 1), np.array([1, 2, 3, 0]).reshape(4, 1),
                                np.array([0, 0, 0, 0]).reshape(4, 1), np.array(["", "", "", ""]).reshape(4, 1)],
    'DateTimeSplitter_Complex': [np.array([1976, 2025, 1976, 2998]).reshape(4, 1), np.array([11, 6, 11, 3]).reshape(4, 1),
                                 np.array([17, 30, 17, 2]).reshape(4, 1), np.array([12, 0, 12, 14]).reshape(4, 1),
                                 np.array([27, 0, 27, 3]).reshape(4, 1), np.array([4, 0, 5, 2]).reshape(4, 1),
                                 np.array([1, 0, 1, 1]).reshape(4, 1), np.array(["", "", "", ""]).reshape(4, 1)],
    'ToKey_SimpleFloat': [np.array([1, 1, 4, 7, 1]).astype(np.float64).reshape(5, 1)],
    'ToKey_SimpleDouble': [np.array([1, 1, 4, 7, 1]).astype(np.double).reshape(5, 1)],
    'ToKey_SimpleString': [np.array(["1", "one", "Hello", "one", "World"]).reshape(5, 1)],
    'ToKey_2col_Double': [np.array([1.0, 1.0, 4.0, 7.0, 1.0]).astype(np.double).reshape(5, 1), np.array([1.0, 2.0, 4.0, 7.0, 2.0]).astype(np.double).reshape(5, 1)],
    'ToKey_2col_Double_String': [np.array([1.0, 1.0, 4.0, 7.0, 1.0]).astype(np.double).reshape(5, 1), np.array(["1", "two", "Hello", "two", "World"]).reshape(5, 1)],
    'ToString_Numbers': [np.array([['4'], ['4'], ['-1'], ['9']]).reshape(4, 1), np.array([['5.000000'], ['5.000000'], ['3.100000'], ['-0.230000']]).reshape(4, 1), np.array([['6.000000'], ['6.700000'], ['NaN'], ['NaN']]).reshape(4, 1)],
    'ToString_Other_Types': [np.array([['True'], ['False']]).reshape(2, 1), np.array([['123.450000'], ['135453984983490.546875']]).reshape(2, 1)],
    'TimeSeriesImputer_1grain_2gap': [np.array([[1], [2], [3], [4], [5], [6], [7]]).astype(np.single).reshape(7, 1), np.array([[10], [11], [12], [12], [13], [13], [14]]).astype(np.single).reshape(7, 1), np.array([[False], [False], [False], [True], [False], [True], [False]]).astype(np.single).reshape(7, 1)],
    'TimeSeriesImputer_1grain_1gap_2filtercolumn': [np.array([[1], [2], [3], [4], [5]]).astype(np.single).reshape(5, 1),
                                                    np.array([[10], [13], [15], [15], [20]]).astype(np.single).reshape(5, 1),
                                                    np.array([[19], [12], [16], [16], [19]]).astype(np.single).reshape(5, 1),
                                                    np.array([[False], [False], [False], [True], [False]]).astype(np.single).reshape(5, 1)],
    'TimeSeriesImputer_2grain_nogap': [np.array([[1], [5], [2], [6]]).astype(np.single).reshape(4, 1), np.array([[1970], [1971], [1970], [1971]]).astype(np.single).reshape(4, 1), np.array([[10], [11], [12], [13]]).astype(np.single).reshape(4, 1), np.array([[False], [False], [False], [False]]).astype(np.single).reshape(4, 1)],
    'TimeSeriesImputer_2grain_2gap': [np.array([[0], [5], [1], [2], [3], [6], [7], [8]]).astype(np.single).reshape(8, 1), np.array([[1970], [1971], [1970], [1970], [1970], [1971], [1971], [1971]]).astype(np.single).reshape(8, 1),
                                      np.array([[10], [11], [10], [10], [12], [11], [11], [13]]).astype(np.single).reshape(8, 1), np.array([[False], [False], [True], [True], [False], [True], [True], [False]]).astype(np.single).reshape(8, 1)],
    'RollingWindow_Simple_Mean': [np.array([[[float("NaN"), float("NaN")]], [[float("NaN"), 1.0]], [[1.0, 1.5]], [[1.5, 2.5]]]).astype(np.single).reshape(4, 1, 2)],
    'RollingWindow_Simple_Min': [np.array([[[float("NaN")]], [[1.0]], [[2.0]], [[3.0]]]).astype(np.single).reshape(4, 1, 1)],
    'RollingWindow_Simple_Max': [np.array([[[float("NaN")]], [[1.0]], [[2.0]], [[3.0]]]).astype(np.single).reshape(4, 1, 1)],
    'RollingWindow_Multiple_Grain': [np.array([[[float("NaN")]], [[1.0]], [[float("NaN")]], [[1.0]]]).astype(np.single).reshape(4, 1, 1)],
    'RollingWindow_NonString_Grain': [np.array([[[float("NaN")]], [[1.0]], [[2.0]], [[3.0]]]).astype(np.single).reshape(4, 1, 1)],
    'LagLead_Simple_Lag': [np.array([[[float("NaN"), float("NaN")], [float("NaN"), float("NaN")]], [[float("NaN"), float("NaN")], [float("NaN"), 1.0]], [[float("NaN"), 1.0], [1.0, 2.0]], [[1.0, 2.0], [2.0, 3.0]]]).astype(np.single).reshape(4, 2, 2)],
    'LagLead_Simple_Lead': [np.array([[[1.0, 2.0], [2.0, 3.0]], [[2.0, 3.0], [3.0, 4.0]], [[3.0, 4.0], [4.0, float("NaN")]], [[4.0, float("NaN")], [float("NaN"), float("NaN")]]]).astype(np.single).reshape(4, 2, 2)],
    'LagLead_Complex': [np.array([[[float("NaN"), float("NaN")], [float("NaN"), float("NaN")], [1.0, 2.0], [2.0, 3.0]], 
                                  [[float("NaN"), float("NaN")], [float("NaN"), 1.0], [2.0, 3.0], [3.0, 4.0]], 
                                  [[float("NaN"), 1.0], [1.0, 2.0], [3.0, 4.0], [4.0, float("NaN")]], 
                                  [[1.0, 2.0], [2.0, 3.0], [4.0, float("NaN")], [float("NaN"), float("NaN")]]]).astype(np.single).reshape(4, 4, 2)],
    'ShortGrainDropper': [np.array([[1.0], [2.0], [3.0]]).reshape(3, 1), np.array([["one"], ["one"], ["one"]]).reshape(3, 1)],
    'RollingWin_Pivot_Integration': [np.array([[3.0], [5.0], [5.0], [7.0], [7.0]]), np.array([['1970'], ['1970'], ['1970'], ['1970'], ['1970']]), np.array([[1.0], [1.0], [2.0], [2.0], [4.0]])],
    'Laglead_Pivot_Integration': [np.array([[3.0], [4.0], [4.0]]), np.array([['one'], ['one'], ['one']]), np.array([[1.0], [1.0], [2.0]]), np.array([[2.0], [2.0], [3.0]])],
}
COL_TO_COMPARE = {
    'DateTimeSplitter_Simple': [1, 2, 3, 4, 5, 6, 7, 8],
    'DateTimeSplitter_Complex': [1, 2, 3, 4, 5, 6, 7, 8],
    'ToKey_SimpleFloat': [0],
    'ToKey_SimpleDouble': [0],
    'ToKey_SimpleString': [0],
    'ToKey_2col_Double': [0, 1],
    'ToKey_2col_Double_String': [0, 1],
    'ToString_Numbers': [3, 4, 5],
    'ToString_Other_Types': [2, 3],
    'TimeSeriesImputer_1grain_2gap': [0, 2, 3],
    'TimeSeriesImputer_1grain_1gap_2filtercolumn': [0, 2, 3, 4],
    'TimeSeriesImputer_2grain_nogap': [0, 1, 2, 3],
    'TimeSeriesImputer_2grain_2gap': [0, 1, 2, 3],
    'RollingWindow_Simple_Mean': [2],
    'RollingWindow_Simple_Min': [2],
    'RollingWindow_Simple_Max': [2],
    'RollingWindow_Multiple_Grain': [2],
    'RollingWindow_NonString_Grain': [2],
    'LagLead_Simple_Lag': [2],
    'LagLead_Simple_Lead': [2],
    'LagLead_Complex': [2],
    'ShortGrainDropper': [0, 1],
    'RollingWin_Pivot_Integration': [0, 1, 3],
    'Laglead_Pivot_Integration': [0, 1, 3, 4],
}

def get_file_size(file_path):
    file_size = 0
    try:
        file_size = os.path.getsize(file_path)
    except:
        pass
    return file_size


def get_tmp_file(suffix=None):
    fd, file_name = tempfile.mkstemp(suffix=suffix)
    fl = os.fdopen(fd, 'w')
    fl.close()
    return file_name


class CaptureOutputContext():
    """
    Context which can be used for
    capturing stdout and stderr. 
    """
    def __enter__(self):
        self.orig_stdout = sys.stdout
        self.orig_stderr = sys.stderr
        self.stdout_capturer = io.StringIO()
        self.stderr_capturer = io.StringIO()
        sys.stdout = self.stdout_capturer
        sys.stderr = self.stderr_capturer
        return self

    def __exit__(self, *args):
        sys.stdout = self.orig_stdout
        sys.stderr = self.orig_stderr
        self.stdout = self.stdout_capturer.getvalue()
        self.stderr = self.stderr_capturer.getvalue()

        if self.stdout:
            print(self.stdout)

        if self.stderr:
            print(self.stderr)

        # free up some memory
        del self.stdout_capturer
        del self.stderr_capturer

def export_to_onnx(estimator, test_case):
    """
    Fit and test an estimator and determine
    if it supports exporting to the ONNX format.
    """
    onnx_path = get_tmp_file('.onnx')
    onnx_json_path = get_tmp_file('.onnx.json')

    output = None
    exported = False
    export_valid = False

    try:
        dataset = TRAINING_DATASETS.get(test_case)
        
        estimator.fit(dataset)

        with CaptureOutputContext() as output:
            estimator.export_to_onnx(onnx_path,
                                     'com.microsoft.ml',
                                     dst_json=onnx_json_path,
                                     onnx_version='Stable')
    except Exception as e:
        print(e)

    onnx_file_size = get_file_size(onnx_path)
    onnx_json_file_size = get_file_size(onnx_json_path)
    
    if (output and
        (onnx_file_size != 0) and
        (onnx_json_file_size != 0) and
        (not 'cannot save itself as ONNX' in output.stdout) and
        (not 'Warning: We do not know how to save the predictor as ONNX' in output.stdout)):

        exported = True

        try:
            sess = rt.InferenceSession(onnx_path)
            names = []
            for input in sess.get_inputs():
                names.append(input.name)
            output_names = []
            for output in sess.get_outputs():
                output_names.append(output.name)

            result = sess.run(None, INFERENCING_DATASETS.get(test_case))
            
            #validate result
            for i in range(0, len(EXPECTED_RESULT.get(test_case))):
                np.testing.assert_array_equal(EXPECTED_RESULT.get(test_case)[i], result[COL_TO_COMPARE.get(test_case)[i]])
            export_valid = True
        except Exception as e:
            print(e)
            
    os.remove(onnx_path)
    os.remove(onnx_json_path)
    return {'exported': exported, 'export_valid': export_valid}

class TestOnnxExport(unittest.TestCase):

    # This method is a static method of the class
    # because there were pytest fixture related
    # issues when the method was in the global scope.
    @staticmethod
    def generate_test_method(test_case):
        def method(self):
            estimator = INSTANCES[test_case]

            result = export_to_onnx(estimator, test_case)
            assert result['exported']
            assert result['export_valid']

        return method

    def test_pivot_one_matrix(self):
            
        df = pd.DataFrame(data=dict(
            colA=[1.0],
            grainA=["one"]
        ))

        xf0 = LagLeadOperator(columns={'colA1': 'colA'},
                           grain_columns=['grainA'], 
                           offsets=[-2, -1, 1],
                           horizon=4)
        binarydata = xf0.fit_transform(df, as_binary_data_stream=True)
        
        xf1 = ForecastingPivot(columns_to_pivot=['colA1'])

        xf1.fit(binarydata)
        onnx_path = get_tmp_file('.onnx')
        onnx_json_path = get_tmp_file('.onnx.json')
        xf1.export_to_onnx(onnx_path,
                            'com.microsoft.ml',
                            dst_json=onnx_json_path,
                            onnx_version='Stable')
        sess = rt.InferenceSession(onnx_path)

        grainA = np.array(["one"]).reshape(1,1)
        colA = np.array([1]).astype(np.double).reshape(1,1)
        colA1 = np.array([1, 4, 6, float("NaN"), 2, 5, float("NaN"), float("NaN"), 3, float("NaN"), float("NaN"), 7]).astype(np.double).reshape(1,3,4)
        pred = sess.run(None, {"grainA":grainA,"colA":colA, "colA1":colA1 })

        expectedHorizon = np.array([4]).astype(np.uint32).reshape(1, 1)
        expectedColA = np.array([1]).astype(np.double).reshape(1, 1)
        expectedLag1 = np.array([1]).astype(np.double).reshape(1, 1)
        expectedLead1 = np.array([2]).astype(np.double).reshape(1, 1)
        expectedLag2 = np.array([3]).astype(np.double).reshape(1, 1)
        
        np.testing.assert_array_equal(pred[0], expectedColA)
        np.testing.assert_array_equal(pred[2], expectedHorizon)
        np.testing.assert_array_equal(pred[3], expectedLag1)
        np.testing.assert_array_equal(pred[4], expectedLead1)
        np.testing.assert_array_equal(pred[5], expectedLag2)
    
    
    def test_pivot_two_matrix(self):
            
        df0 = pd.DataFrame(data=dict(
            colA=[1.0, 2.0],
            grainA=["one", "one"],
            colB=[1.0, 3.0],
            grainB=["one", "one"]
        ))

        xf0 = LagLeadOperator(columns={'colA1': 'colA'},
                           grain_columns=['grainA'], 
                           offsets=[-2, -1, 1],
                           horizon=4)
                           
        xf1 = LagLeadOperator(columns={'colB1': 'colB'},
                           grain_columns=['grainB'], 
                           offsets=[-2, -1],
                           horizon=4)
        xf2 = Pipeline([xf0, xf1])
        binarydata = xf2.fit_transform(df0, as_binary_data_stream=True)

        xf3 = ForecastingPivot(columns_to_pivot=['colA1', 'colB1'])

        xf3.fit(binarydata)

        onnx_path = get_tmp_file('.onnx')
        onnx_json_path = get_tmp_file('.onnx.json')
        xf3.export_to_onnx(onnx_path,
                            'com.microsoft.ml',
                            dst_json=onnx_json_path,
                            onnx_version='Stable')
        sess = rt.InferenceSession(onnx_path)
        colA = np.array([1,2]).astype(np.double).reshape(2,1)
        colB = np.array([1,2]).astype(np.double).reshape(2,1)
        grainA = np.array(["one", "one"]).reshape(2,1)
        grainB = np.array(["one", "one"]).reshape(2,1)

        colA1 = np.array([1, 6, 3, 9, 2, 4, 5, 8, float("NaN"), float("NaN"), 7, 10, 1, 6, 9, 3, 2, 4, 8, 5, float("NaN"), float("NaN"), 10, 7]).astype(np.double).reshape(2,3,4)
        colB1 = np.array([1, float("NaN"), 5, 6, 2, float("NaN"), 3, 4, 1, float("NaN"), 6, 5, 2, float("NaN"), 4, 3]).astype(np.double).reshape(2,2,4)
        pred = sess.run(None, {"colA":colA, "colB":colB, "grainA":grainA, "grainB":grainB, "colA1":colA1, "colB1": colB1 })
        
        expectedColA = np.array([1, 1, 2, 2]).astype(np.double).reshape(4, 1)
        expectedColB = np.array([1, 1, 2, 2]).astype(np.double).reshape(4, 1)
        expectedHorizon = np.array([2, 1, 2, 1]).astype(np.uint32).reshape(4, 1)
        expectedLag1 = np.array([3, 9, 9, 3]).astype(np.double).reshape(4, 1)
        expectedLead1 = np.array([5, 8, 8, 5]).astype(np.double).reshape(4, 1)
        expectedLag2 = np.array([7, 10, 10, 7]).astype(np.double).reshape(4, 1)
        expectedVec2Lag2 = np.array([5, 6, 6, 5]).astype(np.double).reshape(4, 1)
        expectedVec2Lag5 = np.array([3, 4, 4, 3]).astype(np.double).reshape(4, 1)

        np.testing.assert_array_equal(pred[0], expectedColA)
        np.testing.assert_array_equal(pred[2], expectedColB)
        np.testing.assert_array_equal(pred[4], expectedHorizon)
        np.testing.assert_array_equal(pred[5], expectedLag1)
        np.testing.assert_array_equal(pred[6], expectedLead1)
        np.testing.assert_array_equal(pred[7], expectedLag2)
        np.testing.assert_array_equal(pred[8], expectedVec2Lag2)
        np.testing.assert_array_equal(pred[9], expectedVec2Lag5)


for test_case in TEST_CASES:
    test_name = 'test_%s' % test_case.replace('(', '_').replace(')', '').lower()
    
    method = TestOnnxExport.generate_test_method(test_case)
    setattr(TestOnnxExport, test_name, method)

TestOnnxExport.test_pivot_one_matrix
TestOnnxExport.test_pivot_two_matrix

if __name__ == '__main__':
    unittest.main()
