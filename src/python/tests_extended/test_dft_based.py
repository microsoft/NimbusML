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
from nimbusml.preprocessing.schema import ColumnSelector
from nimbusml.preprocessing import ToString, ToKeyImputer, DateTimeSplitter
from nimbusml.timeseries import TimeSeriesImputer, LagLeadOperator, RollingWindow, ForecastingPivot, ShortDrop
from nimbusml.preprocessing import (TensorFlowScorer, FromKey, ToKey,
                                    DateTimeSplitter, OnnxRunner)
import onnxruntime as rt
from data_frame_tool import DataFrameTool as DFT

def get_tmp_file(suffix=None):
    fd, file_name = tempfile.mkstemp(suffix=suffix)
    fl = os.fdopen(fd, 'w')
    fl.close()
    return file_name

class TestAutoMLTransforms(unittest.TestCase):
    

    def test_datetimesplitter(self):
        df = pd.DataFrame(data=dict(
            tokens1=[1, 2, 3, 157161600]
        ))

        cols_to_drop = [
            'dtHour12', 'dtDayOfWeek', 'dtDayOfQuarter',
            'dtDayOfYear', 'dtWeekOfMonth', 'dtQuarterOfYear',
            'dtHalfOfYear', 'dtWeekIso', 'dtYearIso', 'dtMonthLabel',
            'dtAmPmLabel', 'dtDayOfWeekLabel', 'dtIsPaidTimeOff','dtHolidayName'
        ]

        dts = DateTimeSplitter(prefix='dt') << 'tokens1'
        xf = Pipeline([dts, ColumnSelector(drop_columns=cols_to_drop)])
        mlnetresult = xf.fit_transform(df)

        onnx_path = get_tmp_file('.onnx')
        onnx_json_path = get_tmp_file('.onnx.json')
        xf.export_to_onnx(onnx_path,
                            'com.microsoft.ml',
                            dst_json=onnx_json_path,
                            onnx_version='Stable')
                            
        df_tool = DFT(onnx_path)
        result_ort = df_tool.execute(df, [])
        
        if len(result_ort.columns) != len(mlnetresult.columns):
            raise RuntimeError("ERROR: The ORT output does not contain the same number of columns as ML.NET.")
        col_tuples = list(zip(mlnetresult.columns[0:],
                              result_ort.columns[0:]))
        for col_tuple in col_tuples:
            try:
                col_mlnet = mlnetresult.loc[:, col_tuple[0]]
                col_ort = result_ort.loc[:, col_tuple[1]]
                check_kwargs = {
                    'check_names': False,
                    'check_exact': True,
                    'check_dtype': True,
                    'check_less_precise': True
                }
                pd.testing.assert_series_equal(col_mlnet, col_ort, **check_kwargs)
            except Exception as e:
                print(e)
                raise RuntimeError("ERROR: OnnxRunner result does not match ML.NET result.")
            
    def test_datetimesplitter_complex(self):
        df = pd.DataFrame(data=dict(
            tokens1=[217081624, 1751241600, 217081625, 32445842582]
        ))

        cols_to_drop = [
            'dtHour12', 'dtDayOfWeek', 'dtDayOfQuarter',
            'dtDayOfYear', 'dtWeekOfMonth', 'dtQuarterOfYear',
            'dtHalfOfYear', 'dtWeekIso', 'dtYearIso', 'dtMonthLabel',
            'dtAmPmLabel', 'dtDayOfWeekLabel', 'dtIsPaidTimeOff','dtHolidayName'
        ]

        dts = DateTimeSplitter(prefix='dt') << 'tokens1'
        xf = Pipeline([dts, ColumnSelector(drop_columns=cols_to_drop)])
        mlnetresult = xf.fit_transform(df)

        onnx_path = get_tmp_file('.onnx')
        onnx_json_path = get_tmp_file('.onnx.json')
        xf.export_to_onnx(onnx_path,
                            'com.microsoft.ml',
                            dst_json=onnx_json_path,
                            onnx_version='Stable')
        df_tool = DFT(onnx_path)
        result_ort = df_tool.execute(df, [])
        
        if len(result_ort.columns) != len(mlnetresult.columns):
            raise RuntimeError("ERROR: The ORT output does not contain the same number of columns as ML.NET.")
        col_tuples = list(zip(mlnetresult.columns[0:],
                              result_ort.columns[0:]))
        for col_tuple in col_tuples:
            try:
                col_mlnet = mlnetresult.loc[:, col_tuple[0]]
                col_ort = result_ort.loc[:, col_tuple[1]]
                check_kwargs = {
                    'check_names': False,
                    'check_exact': True,
                    'check_dtype': True,
                    'check_less_precise': True
                }
                pd.testing.assert_series_equal(col_mlnet, col_ort, **check_kwargs)
            except Exception as e:
                print(e)
                raise RuntimeError("ERROR: OnnxRunner result does not match ML.NET result.")

    def test_tokey_simple_float(self):
        
        df = pd.DataFrame(data=dict(
            target=[1.0, 1.0, 1.0, 2.0]
        )).astype({'target': np.float64})

        xf = ToKeyImputer()
        mlnetresult = xf.fit_transform(df)

        onnx_path = get_tmp_file('.onnx')
        onnx_json_path = get_tmp_file('.onnx.json')
        xf.export_to_onnx(onnx_path,
                            'com.microsoft.ml',
                            dst_json=onnx_json_path,
                            onnx_version='Stable')
        df_tool = DFT(onnx_path)
        result_ort = df_tool.execute(df, [])
        
        if len(result_ort.columns) != len(mlnetresult.columns):
            raise RuntimeError("ERROR: The ORT output does not contain the same number of columns as ML.NET.")
        col_tuples = list(zip(mlnetresult.columns[0:],
                              result_ort.columns[0:]))
        for col_tuple in col_tuples:
            try:
                col_mlnet = mlnetresult.loc[:, col_tuple[0]]
                col_ort = result_ort.loc[:, col_tuple[1]]
                check_kwargs = {
                    'check_names': False,
                    'check_exact': True,
                    'check_dtype': True,
                    'check_less_precise': True
                }
                pd.testing.assert_series_equal(col_mlnet, col_ort, **check_kwargs)
            except Exception as e:
                print(e)
                raise RuntimeError("ERROR: OnnxRunner result does not match ML.NET result.")
                
    def test_tokey_simple_double(self):
        
        df = pd.DataFrame(data=dict(
            target=[1.0, 1.0, 1.0, 2.0]
        )).astype({'target': np.double})

        xf = ToKeyImputer()
        mlnetresult = xf.fit_transform(df)

        onnx_path = get_tmp_file('.onnx')
        onnx_json_path = get_tmp_file('.onnx.json')
        xf.export_to_onnx(onnx_path,
                            'com.microsoft.ml',
                            dst_json=onnx_json_path,
                            onnx_version='Stable')
        df_tool = DFT(onnx_path)
        result_ort = df_tool.execute(df, [])
        
        if len(result_ort.columns) != len(mlnetresult.columns):
            raise RuntimeError("ERROR: The ORT output does not contain the same number of columns as ML.NET.")
        col_tuples = list(zip(mlnetresult.columns[0:],
                              result_ort.columns[0:]))
        for col_tuple in col_tuples:
            try:
                col_mlnet = mlnetresult.loc[:, col_tuple[0]]
                col_ort = result_ort.loc[:, col_tuple[1]]
                check_kwargs = {
                    'check_names': False,
                    'check_exact': True,
                    'check_dtype': True,
                    'check_less_precise': True
                }
                pd.testing.assert_series_equal(col_mlnet, col_ort, **check_kwargs)
            except Exception as e:
                print(e)
                raise RuntimeError("ERROR: OnnxRunner result does not match ML.NET result.")

    def test_tokey_simple_string(self):
        
        df = pd.DataFrame(data=dict(
            target=["one", "one", "one", "two"]
        ))

        xf = ToKeyImputer()
        mlnetresult = xf.fit_transform(df)

        onnx_path = get_tmp_file('.onnx')
        onnx_json_path = get_tmp_file('.onnx.json')
        xf.export_to_onnx(onnx_path,
                            'com.microsoft.ml',
                            dst_json=onnx_json_path,
                            onnx_version='Stable')
        df_tool = DFT(onnx_path)
        result_ort = df_tool.execute(df, [])
        
        if len(result_ort.columns) != len(mlnetresult.columns):
            raise RuntimeError("ERROR: The ORT output does not contain the same number of columns as ML.NET.")
        col_tuples = list(zip(mlnetresult.columns[0:],
                              result_ort.columns[0:]))
        for col_tuple in col_tuples:
            try:
                col_mlnet = mlnetresult.loc[:, col_tuple[0]]
                col_ort = result_ort.loc[:, col_tuple[1]]
                check_kwargs = {
                    'check_names': False,
                    'check_exact': True,
                    'check_dtype': True,
                    'check_less_precise': True
                }
                pd.testing.assert_series_equal(col_mlnet, col_ort, **check_kwargs)
            except Exception as e:
                print(e)
                raise RuntimeError("ERROR: OnnxRunner result does not match ML.NET result.")

    def test_tokey_2col_double(self):
        
        df = pd.DataFrame(data=dict(
            data1=[1.0, 1.0, 1.0, 2.0],
            data2=[2.0, 2.0, 2.0, 3.0],
        )).astype({'data1': np.double,
                   'data1': np.double})

        xf = ToKeyImputer()
        mlnetresult = xf.fit_transform(df)

        onnx_path = get_tmp_file('.onnx')
        onnx_json_path = get_tmp_file('.onnx.json')
        xf.export_to_onnx(onnx_path,
                            'com.microsoft.ml',
                            dst_json=onnx_json_path,
                            onnx_version='Stable')
        df_tool = DFT(onnx_path)
        result_ort = df_tool.execute(df, [])
        
        if len(result_ort.columns) != len(mlnetresult.columns):
            raise RuntimeError("ERROR: The ORT output does not contain the same number of columns as ML.NET.")
        col_tuples = list(zip(mlnetresult.columns[0:],
                              result_ort.columns[0:]))
        for col_tuple in col_tuples:
            try:
                col_mlnet = mlnetresult.loc[:, col_tuple[0]]
                col_ort = result_ort.loc[:, col_tuple[1]]
                check_kwargs = {
                    'check_names': False,
                    'check_exact': True,
                    'check_dtype': True,
                    'check_less_precise': True
                }
                pd.testing.assert_series_equal(col_mlnet, col_ort, **check_kwargs)
            except Exception as e:
                print(e)
                raise RuntimeError("ERROR: OnnxRunner result does not match ML.NET result.")

    def test_tokey_2col_double_string(self):
        
        df = pd.DataFrame(data=dict(
            data1=[1.0, 1.0, 1.0, 2.0],
            data2=["two", "two", "three", "two"],
        ))

        xf = ToKeyImputer()
        mlnetresult = xf.fit_transform(df)

        onnx_path = get_tmp_file('.onnx')
        onnx_json_path = get_tmp_file('.onnx.json')
        xf.export_to_onnx(onnx_path,
                            'com.microsoft.ml',
                            dst_json=onnx_json_path,
                            onnx_version='Stable')
        df_tool = DFT(onnx_path)
        result_ort = df_tool.execute(df, [])
        
        if len(result_ort.columns) != len(mlnetresult.columns):
            raise RuntimeError("ERROR: The ORT output does not contain the same number of columns as ML.NET.")
        col_tuples = list(zip(mlnetresult.columns[0:],
                              result_ort.columns[0:]))
        for col_tuple in col_tuples:
            try:
                col_mlnet = mlnetresult.loc[:, col_tuple[0]]
                col_ort = result_ort.loc[:, col_tuple[1]]
                check_kwargs = {
                    'check_names': False,
                    'check_exact': True,
                    'check_dtype': True,
                    'check_less_precise': True
                }
                pd.testing.assert_series_equal(col_mlnet, col_ort, **check_kwargs)
            except Exception as e:
                print(e)
                raise RuntimeError("ERROR: OnnxRunner result does not match ML.NET result.")


    def test_tostring_numbers(self):
        data={'f0': [4, 4, -1, 9],
              'f1': [5, 5, 3.1, -0.23],
              'f2': [6, 6.7, np.nan, np.nan]}
        data = pd.DataFrame(data).astype({'f0': np.int32,
                                       'f1': np.float32,
                                       'f2': np.float64})

        xf = ToString(columns={'f0.out': 'f0',
                               'f1.out': 'f1',
                               'f2.out': 'f2'})
        mlnetresult = xf.fit_transform(data)
        
        onnx_path = get_tmp_file('.onnx')
        onnx_json_path = get_tmp_file('.onnx.json')
        xf.export_to_onnx(onnx_path,
                            'com.microsoft.ml',
                            dst_json=onnx_json_path,
                            onnx_version='Stable')
        
        df_tool = DFT(onnx_path)
        result_ort = df_tool.execute(data, [])
        
        if len(result_ort.columns) != len(mlnetresult.columns):
            raise RuntimeError("ERROR: The ORT output does not contain the same number of columns as ML.NET.")
        col_tuples = list(zip(mlnetresult.columns[0:],
                              result_ort.columns[0:]))
        for col_tuple in col_tuples:
            try:
                col_mlnet = mlnetresult.loc[:, col_tuple[0]]
                col_ort = result_ort.loc[:, col_tuple[1]]
                check_kwargs = {
                    'check_names': False,
                    'check_exact': True,
                    'check_dtype': True,
                    'check_less_precise': True
                }
                pd.testing.assert_series_equal(col_mlnet, col_ort, **check_kwargs)
            except Exception as e:
                print(e)
                raise RuntimeError("ERROR: OnnxRunner result does not match ML.NET result.")

    def test_tostring_other_types(self):
        data={'f0': [True, False],
              'f1': [123.45, 135453984983490.5473]}
        data = pd.DataFrame(data).astype({'f0': bool,
                                       'f1': np.double})

        xf = ToString(columns={'f0.out': 'f0',
                               'f1.out': 'f1'})
        mlnetresult = xf.fit_transform(data)
        
        onnx_path = get_tmp_file('.onnx')
        onnx_json_path = get_tmp_file('.onnx.json')
        xf.export_to_onnx(onnx_path,
                            'com.microsoft.ml',
                            dst_json=onnx_json_path,
                            onnx_version='Stable')
        
        df_tool = DFT(onnx_path)
        result_ort = df_tool.execute(data, [])
        
        if len(result_ort.columns) != len(mlnetresult.columns):
            raise RuntimeError("ERROR: The ORT output does not contain the same number of columns as ML.NET.")
        col_tuples = list(zip(mlnetresult.columns[0:],
                              result_ort.columns[0:]))
        for col_tuple in col_tuples:
            try:
                col_mlnet = mlnetresult.loc[:, col_tuple[0]]
                col_ort = result_ort.loc[:, col_tuple[1]]
                check_kwargs = {
                    'check_names': False,
                    'check_exact': True,
                    'check_dtype': True,
                    'check_less_precise': True
                }
                pd.testing.assert_series_equal(col_mlnet, col_ort, **check_kwargs)
            except Exception as e:
                print(e)
                raise RuntimeError("ERROR: OnnxRunner result does not match ML.NET result.")

    def test_timeseriesimputer_onegrain_twogap(self):

        df = pd.DataFrame(data=dict(
            ts=[1, 2, 3, 5, 7],
            grain=[1970, 1970, 1970, 1970, 1970],
            c=[10, 11, 12, 13, 14]
        )).astype({'ts': np.int64, 'grain': np.int32, 'c': np.int32})


        xf = TimeSeriesImputer(time_series_column='ts',
                                filter_columns=['c'],
                                grain_columns=['grain'],
                                impute_mode='ForwardFill',
                                filter_mode='Include')
        mlnetresult = xf.fit_transform(df)

        onnx_path = get_tmp_file('.onnx')
        onnx_json_path = get_tmp_file('.onnx.json')
        xf.export_to_onnx(onnx_path,
                            'com.microsoft.ml',
                            dst_json=onnx_json_path,
                            onnx_version='Stable')

        df_tool = DFT(onnx_path)
        result_ort = df_tool.execute(df, [])
        
        if len(result_ort.columns) != len(mlnetresult.columns):
            raise RuntimeError("ERROR: The ORT output does not contain the same number of columns as ML.NET.")
        col_tuples = list(zip(mlnetresult.columns[0:],
                              result_ort.columns[0:]))
        for col_tuple in col_tuples:
            try:
                col_mlnet = mlnetresult.loc[:, col_tuple[0]]
                col_ort = result_ort.loc[:, col_tuple[1]]
                check_kwargs = {
                    'check_names': False,
                    'check_exact': True,
                    'check_dtype': True,
                    'check_less_precise': True
                }
                pd.testing.assert_series_equal(col_mlnet, col_ort, **check_kwargs)
            except Exception as e:
                print(e)
                raise RuntimeError("ERROR: OnnxRunner result does not match ML.NET result.")

    def test_timeseriesimputer_twograin_nogap(self):

        df = pd.DataFrame(data=dict(
            ts=[1, 5, 2, 6],
            grain=[1970, 1971, 1970, 1971],
            c=[10, 11, 12, 13]
        )).astype({'ts': np.int64, 'grain': np.int32, 'c': np.int32})


        xf = TimeSeriesImputer(time_series_column='ts',
                                filter_columns=['c'],
                                grain_columns=['grain'],
                                impute_mode='ForwardFill',
                                filter_mode='Include')
        mlnetresult = xf.fit_transform(df)

        onnx_path = get_tmp_file('.onnx')
        onnx_json_path = get_tmp_file('.onnx.json')
        xf.export_to_onnx(onnx_path,
                            'com.microsoft.ml',
                            dst_json=onnx_json_path,
                            onnx_version='Stable')

        df_tool = DFT(onnx_path)
        result_ort = df_tool.execute(df, [])
        
        if len(result_ort.columns) != len(mlnetresult.columns):
            raise RuntimeError("ERROR: The ORT output does not contain the same number of columns as ML.NET.")
        col_tuples = list(zip(mlnetresult.columns[0:],
                              result_ort.columns[0:]))
        for col_tuple in col_tuples:
            try:
                col_mlnet = mlnetresult.loc[:, col_tuple[0]]
                col_ort = result_ort.loc[:, col_tuple[1]]
                check_kwargs = {
                    'check_names': False,
                    'check_exact': True,
                    'check_dtype': True,
                    'check_less_precise': True
                }
                pd.testing.assert_series_equal(col_mlnet, col_ort, **check_kwargs)
            except Exception as e:
                print(e)
                raise RuntimeError("ERROR: OnnxRunner result does not match ML.NET result.")

    def test_timeseriesimputer_onegrain_onegap_two_filtercolumn(self):

        df = pd.DataFrame(data=dict(
            ts=[1, 2, 3, 5],
            grain=[1970, 1970, 1970, 1970],
            c3=[10, 13, 15, 20],
            c4=[19, 12, 16, 19]
        )).astype({'ts': np.int64, 'grain': np.int32, 'c3': np.int32, 'c4': np.int32})

        xf = TimeSeriesImputer(time_series_column='ts',
                                grain_columns=['grain'],
                                filter_columns=['c3', 'c4'],
                                impute_mode='ForwardFill',
                                filter_mode='Include')
        mlnetresult = xf.fit_transform(df)

        onnx_path = get_tmp_file('.onnx')
        onnx_json_path = get_tmp_file('.onnx.json')
        xf.export_to_onnx(onnx_path,
                            'com.microsoft.ml',
                            dst_json=onnx_json_path,
                            onnx_version='Stable')

        df_tool = DFT(onnx_path)
        result_ort = df_tool.execute(df, [])
        
        if len(result_ort.columns) != len(mlnetresult.columns):
            raise RuntimeError("ERROR: The ORT output does not contain the same number of columns as ML.NET.")
        col_tuples = list(zip(mlnetresult.columns[0:],
                              result_ort.columns[0:]))
        for col_tuple in col_tuples:
            try:
                col_mlnet = mlnetresult.loc[:, col_tuple[0]]
                col_ort = result_ort.loc[:, col_tuple[1]]
                check_kwargs = {
                    'check_names': False,
                    'check_exact': True,
                    'check_dtype': True,
                    'check_less_precise': True
                }
                pd.testing.assert_series_equal(col_mlnet, col_ort, **check_kwargs)
            except Exception as e:
                print(e)
                raise RuntimeError("ERROR: OnnxRunner result does not match ML.NET result.")

    def test_short_drop(self):

        df = pd.DataFrame(data=dict(
            colA=[1.0, 2.0, 3.0, 4.0],
            grainA=["one", "one", "one", "two"]
        ))

        xf = ShortDrop(columns={'colA1': 'colA'},
                           grain_columns=['grainA'],
                           min_rows=2)
        mlnetresult = xf.fit_transform(df)

        onnx_path = get_tmp_file('.onnx')
        onnx_json_path = get_tmp_file('.onnx.json')
        xf.export_to_onnx(onnx_path,
                            'com.microsoft.ml',
                            dst_json=onnx_json_path,
                            onnx_version='Stable')

        df_tool = DFT(onnx_path)
        result_ort = df_tool.execute(df, [])
        
        if len(result_ort.columns) != len(mlnetresult.columns):
            raise RuntimeError("ERROR: The ORT output does not contain the same number of columns as ML.NET.")
        col_tuples = list(zip(mlnetresult.columns[0:],
                              result_ort.columns[0:]))
        for col_tuple in col_tuples:
            try:
                col_mlnet = mlnetresult.loc[:, col_tuple[0]]
                col_ort = result_ort.loc[:, col_tuple[1]]
                check_kwargs = {
                    'check_names': False,
                    'check_exact': True,
                    'check_dtype': True,
                    'check_less_precise': True
                }
                pd.testing.assert_series_equal(col_mlnet, col_ort, **check_kwargs)
            except Exception as e:
                print(e)
                raise RuntimeError("ERROR: OnnxRunner result does not match ML.NET result.")

    def test_integration_laglead_pivot(self):

        df = pd.DataFrame(data=dict(
            colA=[1.0, 2.0, 3.0, 4.0],
            grainA=["one", "one", "one", "one"]
        ))

        xf0 = LagLeadOperator(columns={'colA1': 'colA'},
                           grain_columns=['grainA'], 
                           offsets=[-2, -1],
                           horizon=2)

        xf1 = ForecastingPivot(columns_to_pivot=['colA1'])

        xf = Pipeline([xf0, xf1])

        mlnetresult = xf.fit_transform(df)
        
        onnx_path = get_tmp_file('.onnx')
        onnx_json_path = get_tmp_file('.onnx.json')
        xf.export_to_onnx(onnx_path,
                            'com.microsoft.ml',
                            dst_json=onnx_json_path,
                            onnx_version='Stable')
                            
        df_tool = DFT(onnx_path)
        result_ort = df_tool.execute(df, [])
        if len(result_ort.columns) != len(mlnetresult.columns):
            raise RuntimeError("ERROR: The ORT output does not contain the same number of columns as ML.NET.")
     
        col_tuples = list(zip(mlnetresult.columns[0:],
                              result_ort.columns[0:]))
        for col_tuple in col_tuples:
            try:
                col_mlnet = mlnetresult.loc[:, col_tuple[0]]
                col_ort = result_ort.loc[:, col_tuple[1]]
                check_kwargs = {
                    'check_names': False,
                    'check_exact': False,
                    'check_dtype': True,
                    'check_less_precise': True
                }
                pd.testing.assert_series_equal(col_mlnet, col_ort, **check_kwargs)
            except Exception as e:
                print(e)
                raise RuntimeError("ERROR: OnnxRunner result does not match ML.NET result.")
                
    def test_integration_rollwin_pivot(self):

        df = pd.DataFrame(data=dict(
            colA=[1.0, 2.0, 3.0, 4.0],
            grainA=["one", "one", "one", "one"]
        ))

        xf0 = RollingWindow(columns={'colA1': 'colA'},
                           grain_column=['grainA'], 
                           window_calculation='Mean',
                           max_window_size=1,
                           horizon=1)

        xf1 = ForecastingPivot(columns_to_pivot=['colA1'])

        xf = Pipeline([xf0, xf1])

        mlnetresult = xf.fit_transform(df)
        
        onnx_path = get_tmp_file('.onnx')
        onnx_json_path = get_tmp_file('.onnx.json')
        xf.export_to_onnx(onnx_path,
                            'com.microsoft.ml',
                            dst_json=onnx_json_path,
                            onnx_version='Stable')
                            
        df_tool = DFT(onnx_path)
        result_ort = df_tool.execute(df, [])
        if len(result_ort.columns) != len(mlnetresult.columns):
            raise RuntimeError("ERROR: The ORT output does not contain the same number of columns as ML.NET.")
     
        col_tuples = list(zip(mlnetresult.columns[0:],
                              result_ort.columns[0:]))
        for col_tuple in col_tuples:
            try:
                col_mlnet = mlnetresult.loc[:, col_tuple[0]]
                col_ort = result_ort.loc[:, col_tuple[1]]
                check_kwargs = {
                    'check_names': False,
                    'check_exact': False,
                    'check_dtype': True,
                    'check_less_precise': True
                }
                pd.testing.assert_series_equal(col_mlnet, col_ort, **check_kwargs)
            except Exception as e:
                print(e)
                raise RuntimeError("ERROR: OnnxRunner result does not match ML.NET result.")

if __name__ == '__main__':
    unittest.main()
