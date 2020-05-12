# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import os
import platform
import tempfile
import unittest

import numpy as np
import pandas as pd
from nimbusml import Pipeline
from nimbusml.preprocessing import ToString, ToKeyImputer, DateTimeSplitter
from nimbusml.preprocessing.schema import ColumnSelector
from nimbusml.timeseries import TimeSeriesImputer, LagLeadOperator, RollingWindow, ForecastingPivot, ShortDrop
from data_frame_tool import DataFrameTool as DFT

def get_tmp_file(suffix=None):
    fd, file_name = tempfile.mkstemp(suffix=suffix)
    fl = os.fdopen(fd, 'w')
    fl.close()
    return file_name

class TestAutoMLTransforms(unittest.TestCase):

    def test_tostring(self):
        data={'f0': [4, 4, -1, 9],
              'f1': [5, 5, 3.1, -0.23],
              'f2': [6, 6.7, np.nan, np.nan]}
        data = pd.DataFrame(data).astype({'f0': np.int32,
                                       'f1': np.float32,
                                       'f2': np.float64})

        xf = ToString(columns={'f0.out': 'f0',
                               'f1.out': 'f1',
                               'f2.out': 'f2'})
        xf.fit(data)

        onnx_path = get_tmp_file('.onnx')
        onnx_json_path = get_tmp_file('.onnx.json')
        xf.export_to_onnx(onnx_path,
                            'com.microsoft.ml',
                            dst_json=onnx_json_path,
                            onnx_version='Stable')

    def test_tokeyimputer(self):
        text_df = pd.DataFrame(
            data=dict(
                text=[
                    "cat",
                    "dog",
                    "fish",
                    "orange",
                    "cat orange",
                    "dog",
                    "fish",
                    None,
                    "spider"]))

        xf = ToKeyImputer() << 'text'
        xf.fit(text_df)

        onnx_path = get_tmp_file('.onnx')
        onnx_json_path = get_tmp_file('.onnx.json')
        xf.export_to_onnx(onnx_path,
                            'com.microsoft.ml',
                            dst_json=onnx_json_path,
                            onnx_version='Stable')

    def test_datetimesplitter(self):
        df = pd.DataFrame(data=dict(
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
        xf = Pipeline([dts, ColumnSelector(drop_columns=cols_to_drop)])
        xf.fit(df)

        onnx_path = get_tmp_file('.onnx')
        onnx_json_path = get_tmp_file('.onnx.json')
        xf.export_to_onnx(onnx_path,
                            'com.microsoft.ml',
                            dst_json=onnx_json_path,
                            onnx_version='Stable')

    def test_timeseriesimputer(self):

        df = pd.DataFrame(data=dict(
            ts=[1, 2, 3, 5],
            grain=[1970, 1970, 1970, 1970],
            c3=[10, 13, 15, 20],
            c4=[19, 12, 16, 19]
        ))

        xf = TimeSeriesImputer(time_series_column='ts',
                                grain_columns=['grain'],
                                filter_columns=['c3', 'c4'],
                                impute_mode='ForwardFill',
                                filter_mode='Include')
        xf.fit(df)

        onnx_path = get_tmp_file('.onnx')
        onnx_json_path = get_tmp_file('.onnx.json')
        xf.export_to_onnx(onnx_path,
                            'com.microsoft.ml',
                            dst_json=onnx_json_path,
                            onnx_version='Stable')

    def test_shortdrop(self):

        df = pd.DataFrame(data=dict(
            ts=[1.0, 3.0, 5.0, 7.0],
            grain=['1970', '1970', '1970', '1970'],
        ))

        xf = ShortDrop(grain_columns=['grain'], min_rows=4) << 'ts'
        xf.fit(df)

        onnx_path = get_tmp_file('.onnx')
        onnx_json_path = get_tmp_file('.onnx.json')
        xf.export_to_onnx(onnx_path,
                            'com.microsoft.ml',
                            dst_json=onnx_json_path,
                            onnx_version='Stable')

    def test_rolling_window(self):

        df = pd.DataFrame(data=dict(
            ts=[1.0, 3.0, 5.0, 7.0],
            grain=['1970', '1970', '1970', '1970'],
        ))

        xf = RollingWindow(columns={'ts_r': 'ts'},
                           grain_columns=['grain'], 
                           window_calculation='Mean',
                           max_window_size=1,
                           horizon=2)
        xf.fit(df)

        onnx_path = get_tmp_file('.onnx')
        onnx_json_path = get_tmp_file('.onnx.json')
        xf.export_to_onnx(onnx_path,
                            'com.microsoft.ml',
                            dst_json=onnx_json_path,
                            onnx_version='Stable')

    def test_pivot(self):

        df = pd.DataFrame(data=dict(
            ts=[1.0, 3.0, 5.0, 7.0],
            grain=['1970', '1970', '1970', '1970'],
        ))

        xf0 = RollingWindow(columns={'ts_r': 'ts'},
                           grain_columns=['grain'], 
                           window_calculation='Mean',
                           max_window_size=1,
                           horizon=1)

        xf1 = ForecastingPivot(columns_to_pivot=['ts_r'])

        xf = Pipeline([xf0, xf1])

        xf.fit(df)

        onnx_path = get_tmp_file('.onnx')
        onnx_json_path = get_tmp_file('.onnx.json')
        xf.export_to_onnx(onnx_path,
                            'com.microsoft.ml',
                            dst_json=onnx_json_path,
                            onnx_version='Stable')

    def test_lag(self):

        df = pd.DataFrame(data=dict(
            ts=[1.0, 3.0, 5.0, 7.0],
            grain=['1970', '1970', '1970', '1970'],
        ))

        xf = LagLeadOperator(columns={'ts_r': 'ts'}, 
                           grain_columns=['grain'], 
                           offsets=[-1, 1],
                           horizon=1)

        xf.fit(df)
        onnx_path = get_tmp_file('.onnx')
        onnx_json_path = get_tmp_file('.onnx.json')
        xf.export_to_onnx(onnx_path,
                            'com.microsoft.ml',
                            dst_json=onnx_json_path,
                            onnx_version='Stable')

if __name__ == '__main__':
    unittest.main()
