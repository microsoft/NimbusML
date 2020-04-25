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
    # invalid input test
    
    def test_datetimesplitter_bad_input_data(self):
        df = pd.DataFrame(data=dict(
            tokens1=[1, 2, 3, 157161600]
        ))

        cols_to_drop = [
            'dtHour12', 'dtDayOfWeek', 'dtDayOfQuarter',
            'dtDayOfYear', 'dtWeekOfMonth', 'dtQuarterOfYear',
            'dtHalfOfYear', 'dtWeekIso', 'dtYearIso', 'dtMonthLabel',
            'dtAmPmLabel', 'dtDayOfWeekLabel', 'dtIsPaidTimeOff'
        ]

        dts = DateTimeSplitter(prefix='dt') << 'tokens1'
        xf = Pipeline([dts, ColumnSelector(drop_columns=cols_to_drop)])
        xf.fit(df)

        onnx_path = get_tmp_file('.onnx')
        onnx_json_path = get_tmp_file('.onnx.json')
        xf.export_to_onnx(onnx_path,
                            'com.microsoft.ml',
                            dst_json=onnx_json_path,
                            onnx_version='Stable')
        sess = rt.InferenceSession(onnx_path)

        with self.assertRaisesRegex(Exception, "ONNXRuntimeError"):
            data1 = np.array([1, 2, 3, -3]).reshape(4,1)
            pred = sess.run(None, {"tokens1":data1})
            
    def test_datetimesplitter_bad_input_type(self):
        df = pd.DataFrame(data=dict(
            tokens1=[1, 2, 3, 157161600]
        ))

        cols_to_drop = [
            'dtHour12', 'dtDayOfWeek', 'dtDayOfQuarter',
            'dtDayOfYear', 'dtWeekOfMonth', 'dtQuarterOfYear',
            'dtHalfOfYear', 'dtWeekIso', 'dtYearIso', 'dtMonthLabel',
            'dtAmPmLabel', 'dtDayOfWeekLabel', 'dtIsPaidTimeOff'
        ]

        dts = DateTimeSplitter(prefix='dt') << 'tokens1'
        xf = Pipeline([dts, ColumnSelector(drop_columns=cols_to_drop)])
        xf.fit(df)

        onnx_path = get_tmp_file('.onnx')
        onnx_json_path = get_tmp_file('.onnx.json')
        xf.export_to_onnx(onnx_path,
                            'com.microsoft.ml',
                            dst_json=onnx_json_path,
                            onnx_version='Stable')
        sess = rt.InferenceSession(onnx_path)

        with self.assertRaisesRegex(Exception, "ONNXRuntimeError"):
            data1 = np.array([1, 2, 3, "3"]).reshape(4,1)
            pred = sess.run(None, {"tokens1":data1})

    def test_datetimesplitter_bad_shape(self):
        df = pd.DataFrame(data=dict(
            tokens1=[1, 2, 3, 157161600]
        ))

        cols_to_drop = [
            'dtHour12', 'dtDayOfWeek', 'dtDayOfQuarter',
            'dtDayOfYear', 'dtWeekOfMonth', 'dtQuarterOfYear',
            'dtHalfOfYear', 'dtWeekIso', 'dtYearIso', 'dtMonthLabel',
            'dtAmPmLabel', 'dtDayOfWeekLabel', 'dtIsPaidTimeOff'
        ]

        dts = DateTimeSplitter(prefix='dt') << 'tokens1'
        xf = Pipeline([dts, ColumnSelector(drop_columns=cols_to_drop)])
        xf.fit(df)

        onnx_path = get_tmp_file('.onnx')
        onnx_json_path = get_tmp_file('.onnx.json')
        xf.export_to_onnx(onnx_path,
                            'com.microsoft.ml',
                            dst_json=onnx_json_path,
                            onnx_version='Stable')
        sess = rt.InferenceSession(onnx_path)

        with self.assertRaisesRegex(Exception, "ONNXRuntimeError"):
            data1 = np.array([[1, 2, 3, 3], [1, 2, 3, 4]]).reshape(4,2)
            pred = sess.run(None, {"tokens1":data1})
    

    def test_tokey_bad_input_type(self):
        
        df = pd.DataFrame(data=dict(
            target=[1.0, 1.0, 1.0, 2.0]
        )).astype({'target': np.float64})

        xf = ToKeyImputer()
        xf.fit(df)

        onnx_path = get_tmp_file('.onnx')
        onnx_json_path = get_tmp_file('.onnx.json')
        xf.export_to_onnx(onnx_path,
                            'com.microsoft.ml',
                            dst_json=onnx_json_path,
                            onnx_version='Stable')
        sess = rt.InferenceSession(onnx_path)

        with self.assertRaisesRegex(Exception, "ONNXRuntimeError"):
            data = np.array([1, float("NaN"), "", 7, float("NaN")]).reshape(5,1)
            pred = sess.run(None, {"target":data})

    def test_tokey_bad_shape(self):
        
        df = pd.DataFrame(data=dict(
            target=[1.0, 1.0, 1.0, 2.0]
        )).astype({'target': np.float64})

        xf = ToKeyImputer()
        xf.fit(df)

        onnx_path = get_tmp_file('.onnx')
        onnx_json_path = get_tmp_file('.onnx.json')
        xf.export_to_onnx(onnx_path,
                            'com.microsoft.ml',
                            dst_json=onnx_json_path,
                            onnx_version='Stable')
        sess = rt.InferenceSession(onnx_path)

        with self.assertRaisesRegex(Exception, "ONNXRuntimeError"):
            data = np.array([[1, float("NaN")], [1, float("NaN")]]).reshape(2,2)
            pred = sess.run(None, {"target":data})

    def test_tostring_bad_input_type(self):
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
        
        # testing
        sess = rt.InferenceSession(onnx_path)
 
        with self.assertRaisesRegex(Exception, "ONNXRuntimeError"):
            f0 = np.array([4, 4, -1, 9]).astype(np.int32).reshape(4,1)
            f1 = np.array([5, 5, 3.1, -0.23]).astype(np.float32).reshape(4,1)
            f2 = np.array([6, "6.7", float("NaN"), float("NaN")]).reshape(4,1)

            # Run your inference session with your model and your data
            pred_list = sess.run(None, {"f0":f0, "f1": f1, "f2":f2  })

    def test_tostring_bad_shape(self):
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
        
        # testing
        sess = rt.InferenceSession(onnx_path)
 
        with self.assertRaisesRegex(Exception, "ONNXRuntimeError"):
            f0 = np.array([[4, 4, -1, 9],[4, 4, -1, 9]]).astype(np.int32).reshape(4,2)
            f1 = np.array([5, 5, 3.1, -0.23]).astype(np.float32).reshape(4,1)
            f2 = np.array([6, 6.7, float("NaN"), float("NaN")]).reshape(4,1)

            # Run your inference session with your model and your data
            pred_list = sess.run(None, {"f0":f0, "f1": f1, "f2":f2  })

    def test_timeseriesimputer_bad_type(self):

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
        xf.fit(df)
        
        onnx_path = get_tmp_file('.onnx')
        onnx_json_path = get_tmp_file('.onnx.json')
        xf.export_to_onnx(onnx_path,
                            'com.microsoft.ml',
                            dst_json=onnx_json_path,
                            onnx_version='Stable')
             
        # testing
        sess = rt.InferenceSession(onnx_path)

        with self.assertRaisesRegex(Exception, "ONNXRuntimeError"):
            ts = np.array([1, 2, 3, 5, 7]).astype(np.int64).reshape(5,1)
            grain = np.array([1970, "1970", 1970, 1970, 1970]).reshape(5,1)
            c = np.array([10, 11, 12, 13, 14]).astype(np.int32).reshape(5,1)
 
            # Run your inference session with your model and your data
            pred_list = sess.run(None, {"ts":ts, "grain": grain, 'c':c})


    def test_timeseriesimputer_bad_input(self):

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
        xf.fit(df)
        
        onnx_path = get_tmp_file('.onnx')
        onnx_json_path = get_tmp_file('.onnx.json')
        xf.export_to_onnx(onnx_path,
                            'com.microsoft.ml',
                            dst_json=onnx_json_path,
                            onnx_version='Stable')
             
        # testing
        sess = rt.InferenceSession(onnx_path)

        with self.assertRaisesRegex(Exception, "ONNXRuntimeError"):
            ts = np.array([1, 2, 3, -5, 7]).astype(np.int64).reshape(5,1)
            grain = np.array([1970, 1970, 1970, 1970, 1970]).reshape(5,1)
            c = np.array([10, 11, 12, 13, 14]).astype(np.int32).reshape(5,1)
 
            # Run your inference session with your model and your data
            pred_list = sess.run(None, {"ts":ts, "grain": grain, 'c':c})

    def test_timeseriesimputer_bad_shape(self):

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
        xf.fit(df)
        
        onnx_path = get_tmp_file('.onnx')
        onnx_json_path = get_tmp_file('.onnx.json')
        xf.export_to_onnx(onnx_path,
                            'com.microsoft.ml',
                            dst_json=onnx_json_path,
                            onnx_version='Stable')
             
        # testing
        sess = rt.InferenceSession(onnx_path)

        with self.assertRaisesRegex(Exception, "ONNXRuntimeError"):
            ts = np.array([[1, 2, 3, 5, 7], [1, 2, 3, 5, 7]]).astype(np.int64).reshape(5,2)
            grain = np.array([1970, 1970, 1970, 1970, 1970]).reshape(5,1)
            c = np.array([10, 11, 12, 13, 14]).astype(np.int32).reshape(5,1)
 
            # Run your inference session with your model and your data
            pred_list = sess.run(None, {"ts":ts, "grain": grain, 'c':c})



    def test_rolling_window_bad_input_type(self):

        df = pd.DataFrame(data=dict(
            colA=[1.0, 2.0, 3.0, 4.0],
            grainA=["one", "one", "one", "one"]
        ))

        xf = RollingWindow(columns={'colA1': 'colA'},
                           grain_column=['grainA'], 
                           window_calculation='Mean',
                           max_window_size=2,
                           horizon=2)
        xf.fit(df)

        onnx_path = get_tmp_file('.onnx')
        onnx_json_path = get_tmp_file('.onnx.json')
        xf.export_to_onnx(onnx_path,
                            'com.microsoft.ml',
                            dst_json=onnx_json_path,
                            onnx_version='Stable')

        # testing
        sess = rt.InferenceSession(onnx_path)
        
        with self.assertRaisesRegex(Exception, "ONNXRuntimeError"):
            grainA = np.array(["one", "one", "one", "one"]).reshape(4,1)
            colA = np.array([1.0, 2.0, 3.0, "4.0"]).reshape(4,1)
    
            # Run your inference session with your model and your data
            pred_list = sess.run(None, {"grainA":grainA, "colA": colA  })


    def test_rolling_window_bad_shape(self):

        df = pd.DataFrame(data=dict(
            colA=[1.0, 2.0, 3.0, 4.0],
            grainA=["one", "one", "one", "one"]
        ))

        xf = RollingWindow(columns={'colA1': 'colA'},
                           grain_column=['grainA'], 
                           window_calculation='Mean',
                           max_window_size=2,
                           horizon=2)
        xf.fit(df)

        onnx_path = get_tmp_file('.onnx')
        onnx_json_path = get_tmp_file('.onnx.json')
        xf.export_to_onnx(onnx_path,
                            'com.microsoft.ml',
                            dst_json=onnx_json_path,
                            onnx_version='Stable')

        # testing
        sess = rt.InferenceSession(onnx_path)
        
        with self.assertRaisesRegex(Exception, "ONNXRuntimeError"):
            grainA = np.array(["one", "one", "one", "one"]).reshape(4,1)
            colA = np.array([[1.0, 2.0, 3.0, 4.0],[1.0, 2.0, 3.0, 4.0]]).reshape(4,2)
    
            # Run your inference session with your model and your data
            pred_list = sess.run(None, {"grainA":grainA, "colA": colA  })

    def test_laglead_bad_input_type(self):

        df = pd.DataFrame(data=dict(
            colA=[1.0, 2.0, 3.0, 4.0],
            grainA=["one", "one", "one", "one"]
        ))

        xf = LagLeadOperator(columns={'colA1': 'colA'},
                           grain_columns=['grainA'], 
                           offsets=[-1],
                           horizon=1)
        xf.fit(df)

        onnx_path = get_tmp_file('.onnx')
        onnx_json_path = get_tmp_file('.onnx.json')
        xf.export_to_onnx(onnx_path,
                            'com.microsoft.ml',
                            dst_json=onnx_json_path,
                            onnx_version='Stable')
        
        # testing
        sess = rt.InferenceSession(onnx_path)
        
        with self.assertRaisesRegex(Exception, "ONNXRuntimeError"):
            grainA = np.array(["one", "one", "one", "one"]).reshape(4,1)
            colA = np.array([1, 2, 3, "4"]).reshape(4,1)
    
            # Run your inference session with your model and your data
            pred_list = sess.run(None, {"grainA":grainA, "colA": colA  })
            
            
    def test_laglead_bad_shape(self):

        df = pd.DataFrame(data=dict(
            colA=[1.0, 2.0, 3.0, 4.0],
            grainA=["one", "one", "one", "one"]
        ))

        xf = LagLeadOperator(columns={'colA1': 'colA'},
                           grain_columns=['grainA'], 
                           offsets=[-1],
                           horizon=1)
        xf.fit(df)

        onnx_path = get_tmp_file('.onnx')
        onnx_json_path = get_tmp_file('.onnx.json')
        xf.export_to_onnx(onnx_path,
                            'com.microsoft.ml',
                            dst_json=onnx_json_path,
                            onnx_version='Stable')
        
        # testing
        sess = rt.InferenceSession(onnx_path)
        
        with self.assertRaisesRegex(Exception, "ONNXRuntimeError"):
            grainA = np.array(["one", "one", "one", "one"]).reshape(4,1)
            colA = np.array([[1.0, 2.0, 3.0, 4.0],[1.0, 2.0, 3.0, 4.0]]).reshape(4,2)
    
            # Run your inference session with your model and your data
            pred_list = sess.run(None, {"grainA":grainA, "colA": colA  })

            
    def test_short_drop_bad_input_type(self):

        df = pd.DataFrame(data=dict(
            colA=[1.0, 2.0, 3.0, 4.0],
            grainA=["one", "one", "one", "two"]
        ))

        xf = ShortDrop(columns={'colA1': 'colA'},
                           grain_columns=['grainA'],
                           min_rows=2)
        xf.fit(df)

        onnx_path = get_tmp_file('.onnx')
        onnx_json_path = get_tmp_file('.onnx.json')
        xf.export_to_onnx(onnx_path,
                            'com.microsoft.ml',
                            dst_json=onnx_json_path,
                            onnx_version='Stable')

        # testing
        sess = rt.InferenceSession(onnx_path)
 
        with self.assertRaisesRegex(Exception, "ONNXRuntimeError"):
            grainA = np.array(["one", "one", "one", "two"]).reshape(4,1)
            colA = np.array([1, 2, 3, "4"]).reshape(4,1)
    
            # Run your inference session with your model and your data
            pred_list = sess.run(None, {"grainA":grainA, "colA": colA  })
            
    def test_short_drop_bad_shape(self):

        df = pd.DataFrame(data=dict(
            colA=[1.0, 2.0, 3.0, 4.0],
            grainA=["one", "one", "one", "two"]
        ))

        xf = ShortDrop(columns={'colA1': 'colA'},
                           grain_columns=['grainA'],
                           min_rows=2)
        xf.fit(df)

        onnx_path = get_tmp_file('.onnx')
        onnx_json_path = get_tmp_file('.onnx.json')
        xf.export_to_onnx(onnx_path,
                            'com.microsoft.ml',
                            dst_json=onnx_json_path,
                            onnx_version='Stable')

        # testing
        sess = rt.InferenceSession(onnx_path)
 
        with self.assertRaisesRegex(Exception, "ONNXRuntimeError"):
            grainA = np.array(["one", "one", "one", "two"]).reshape(4,1)
            colA = np.array([[1.0, 2.0, 3.0, 4.0],[1.0, 2.0, 3.0, 4.0]]).reshape(4,2)
    
            # Run your inference session with your model and your data
            pred_list = sess.run(None, {"grainA":grainA, "colA": colA  })
            
    def test_pivot_bad_input_type(self):
            
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

        with self.assertRaisesRegex(Exception, "ONNXRuntimeError"):
            grainA = np.array(["one"]).reshape(1,1)
            colA = np.array([1]).astype(np.double).reshape(1,1)
            colA1 = np.array([1, 4, "6", float("NaN"), 2, 5, float("NaN"), float("NaN"), 3, float("NaN"), float("NaN"), 7]).reshape(1,3,4)
            pred = sess.run(None, {"grainA":grainA,"colA":colA, "colA1":colA1 })
            

    def test_pivot_bad_shape(self):
            
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

        with self.assertRaisesRegex(Exception, "ONNXRuntimeError"):
            grainA = np.array(["one"]).reshape(1,1)
            colA = np.array([1]).astype(np.double).reshape(1,1)
            colA1 = np.array([1, 4, 6, float("NaN"), 2, 5, float("NaN"), float("NaN"), 3, float("NaN"), float("NaN"), 7]).reshape(1,2,6)
            pred = sess.run(None, {"grainA":grainA,"colA":colA, "colA1":colA1 })

    # normal tests
    def test_datetimesplitter(self):
        df = pd.DataFrame(data=dict(
            tokens1=[1, 2, 3, 157161600]
        ))

        cols_to_drop = [
            'dtHour12', 'dtDayOfWeek', 'dtDayOfQuarter',
            'dtDayOfYear', 'dtWeekOfMonth', 'dtQuarterOfYear',
            'dtHalfOfYear', 'dtWeekIso', 'dtYearIso', 'dtMonthLabel',
            'dtAmPmLabel', 'dtDayOfWeekLabel', 'dtIsPaidTimeOff'
        ]

        dts = DateTimeSplitter(prefix='dt') << 'tokens1'
        xf = Pipeline([dts, ColumnSelector(drop_columns=cols_to_drop)])
        xf.fit(df)

        onnx_path = get_tmp_file('.onnx')
        onnx_json_path = get_tmp_file('.onnx.json')
        xf.export_to_onnx(onnx_path,
                            'com.microsoft.ml',
                            dst_json=onnx_json_path,
                            onnx_version='Stable')
        sess = rt.InferenceSession(onnx_path)

        data1 = np.array([1, 2, 3, 157161600]).astype(np.int64).reshape(4,1)
        pred = sess.run(None, {"tokens1":data1})
        
        expected_years = np.array([1970, 1970, 1970, 1974]).reshape(4, 1)
        expected_month = np.array([1, 1, 1, 12]).reshape(4, 1)
        expected_day = np.array([1, 1, 1, 25]).reshape(4, 1)
        expected_hour = np.array([0, 0, 0, 0]).reshape(4, 1)
        expected_minute = np.array([0, 0, 0, 0]).reshape(4, 1)
        expected_second = np.array([1, 2, 3, 0]).reshape(4, 1)
        expected_ampm = np.array([0, 0, 0, 0]).reshape(4, 1)
        expected_holidayname = np.array(["", "", "", ""]).reshape(4, 1)
        
        np.testing.assert_array_equal(pred[1],expected_years)
        np.testing.assert_array_equal(pred[2],expected_month)
        np.testing.assert_array_equal(pred[3],expected_day)
        np.testing.assert_array_equal(pred[4],expected_hour)
        np.testing.assert_array_equal(pred[5],expected_minute)
        np.testing.assert_array_equal(pred[6],expected_second)
        np.testing.assert_array_equal(pred[7],expected_ampm)
        np.testing.assert_array_equal(pred[8],expected_holidayname)
            
    def test_datetimesplitter_complex(self):
        df = pd.DataFrame(data=dict(
            tokens1=[217081624, 1751241600, 217081625, 32445842582]
        ))

        cols_to_drop = [
            'dtHour12', 'dtDayOfWeek', 'dtDayOfQuarter',
            'dtDayOfYear', 'dtWeekOfMonth', 'dtQuarterOfYear',
            'dtHalfOfYear', 'dtWeekIso', 'dtYearIso', 'dtMonthLabel',
            'dtAmPmLabel', 'dtDayOfWeekLabel', 'dtIsPaidTimeOff'
        ]

        dts = DateTimeSplitter(prefix='dt') << 'tokens1'
        xf = Pipeline([dts, ColumnSelector(drop_columns=cols_to_drop)])
        xf.fit(df)

        onnx_path = get_tmp_file('.onnx')
        onnx_json_path = get_tmp_file('.onnx.json')
        xf.export_to_onnx(onnx_path,
                            'com.microsoft.ml',
                            dst_json=onnx_json_path,
                            onnx_version='Stable')
        sess = rt.InferenceSession(onnx_path)

        data1 = np.array([217081624, 1751241600, 217081625, 32445842582]).astype(np.int64).reshape(4,1)
        pred = sess.run(None, {"tokens1":data1})
        
        expected_years = np.array([1976, 2025, 1976, 2998]).reshape(4, 1)
        expected_month = np.array([11, 6, 11, 3]).reshape(4, 1)
        expected_day = np.array([17, 30, 17, 2]).reshape(4, 1)
        expected_hour = np.array([12, 0, 12, 14]).reshape(4, 1)
        expected_minute = np.array([27, 0, 27, 3]).reshape(4, 1)
        expected_second = np.array([4, 0, 5, 2]).reshape(4, 1)
        expected_ampm = np.array([1, 0, 1, 1]).reshape(4, 1)
        expected_holidayname = np.array(["", "", "", ""]).reshape(4, 1)
        
        np.testing.assert_array_equal(pred[1],expected_years)
        np.testing.assert_array_equal(pred[2],expected_month)
        np.testing.assert_array_equal(pred[3],expected_day)
        np.testing.assert_array_equal(pred[4],expected_hour)
        np.testing.assert_array_equal(pred[5],expected_minute)
        np.testing.assert_array_equal(pred[6],expected_second)
        np.testing.assert_array_equal(pred[7],expected_ampm)
        np.testing.assert_array_equal(pred[8],expected_holidayname)

    def test_tokey_simple_float(self):
        
        df = pd.DataFrame(data=dict(
            target=[1.0, 1.0, 1.0, 2.0]
        )).astype({'target': np.float64})

        xf = ToKeyImputer()
        xf.fit(df)

        onnx_path = get_tmp_file('.onnx')
        onnx_json_path = get_tmp_file('.onnx.json')
        xf.export_to_onnx(onnx_path,
                            'com.microsoft.ml',
                            dst_json=onnx_json_path,
                            onnx_version='Stable')
        sess = rt.InferenceSession(onnx_path)

        data = np.array([1, float("NaN"), 4, 7, float("NaN")]).astype(np.float64).reshape(5,1)
        pred = sess.run(None, {"target":data})

        expectedData = np.array([1, 1, 4, 7, 1]).astype(np.float64).reshape(5, 1)
        
        np.testing.assert_array_equal(pred[0],expectedData)

    def test_tokey_simple_double(self):
        
        df = pd.DataFrame(data=dict(
            target=[1.0, 1.0, 1.0, 2.0]
        )).astype({'target': np.double})

        xf = ToKeyImputer()
        xf.fit(df)

        onnx_path = get_tmp_file('.onnx')
        onnx_json_path = get_tmp_file('.onnx.json')
        xf.export_to_onnx(onnx_path,
                            'com.microsoft.ml',
                            dst_json=onnx_json_path,
                            onnx_version='Stable')
        sess = rt.InferenceSession(onnx_path)

        data = np.array([1, float("NaN"), 4, 7, float("NaN")]).astype(np.double).reshape(5,1)
        pred = sess.run(None, {"target":data})

        expectedData = np.array([1, 1, 4, 7, 1]).astype(np.double).reshape(5, 1)
        
        np.testing.assert_array_equal(pred[0],expectedData)

    def test_tokey_simple_string(self):
        
        df = pd.DataFrame(data=dict(
            target=["one", "one", "one", "two"]
        ))

        xf = ToKeyImputer()
        xf.fit(df)

        onnx_path = get_tmp_file('.onnx')
        onnx_json_path = get_tmp_file('.onnx.json')
        xf.export_to_onnx(onnx_path,
                            'com.microsoft.ml',
                            dst_json=onnx_json_path,
                            onnx_version='Stable')
        sess = rt.InferenceSession(onnx_path)

        data = np.array(["1", "", "Hello", "", "World"]).reshape(5,1)
        pred = sess.run(None, {"target":data})

        expectedData = np.array(["1", "one", "Hello", "one", "World"]).reshape(5, 1)
        
        np.testing.assert_array_equal(pred[0],expectedData)

    def test_tokey_2col_double(self):
        
        df = pd.DataFrame(data=dict(
            data1=[1.0, 1.0, 1.0, 2.0],
            data2=[2.0, 2.0, 2.0, 3.0],
        )).astype({'data1': np.double,
                   'data1': np.double})

        xf = ToKeyImputer()
        xf.fit(df)

        onnx_path = get_tmp_file('.onnx')
        onnx_json_path = get_tmp_file('.onnx.json')
        xf.export_to_onnx(onnx_path,
                            'com.microsoft.ml',
                            dst_json=onnx_json_path,
                            onnx_version='Stable')
        sess = rt.InferenceSession(onnx_path)

        data1 = np.array([1.0, float("NaN"), 4.0, 7.0, float("NaN")]).astype(np.double).reshape(5,1)
        data2 = np.array([1.0, float("NaN"), 4.0, 7.0, float("NaN")]).astype(np.double).reshape(5,1)
        pred = sess.run(None, {"data1":data1, "data2":data2})

        expectedData1 = np.array([1.0, 1.0, 4.0, 7.0, 1.0]).astype(np.double).reshape(5, 1)
        expectedData2 = np.array([1.0, 2.0, 4.0, 7.0, 2.0]).astype(np.double).reshape(5, 1)
        
        np.testing.assert_array_equal(pred[0],expectedData1)
        np.testing.assert_array_equal(pred[1],expectedData2)

    def test_tokey_2col_double_string(self):
        
        df = pd.DataFrame(data=dict(
            data1=[1.0, 1.0, 1.0, 2.0],
            data2=["two", "two", "three", "two"],
        ))

        xf = ToKeyImputer()
        xf.fit(df)

        onnx_path = get_tmp_file('.onnx')
        onnx_json_path = get_tmp_file('.onnx.json')
        xf.export_to_onnx(onnx_path,
                            'com.microsoft.ml',
                            dst_json=onnx_json_path,
                            onnx_version='Stable')
        sess = rt.InferenceSession(onnx_path)

        data1 = np.array([1.0, float("NaN"), 4.0, 7.0, float("NaN")]).astype(np.double).reshape(5,1)
        data2 = np.array(["1", "", "Hello", "", "World"]).reshape(5,1)
        pred = sess.run(None, {"data1":data1, "data2":data2})

        expectedData1 = np.array([1.0, 1.0, 4.0, 7.0, 1.0]).astype(np.double).reshape(5, 1)
        expectedData2 = np.array(["1", "two", "Hello", "two", "World"]).reshape(5, 1)
        
        np.testing.assert_array_equal(pred[0],expectedData1)
        np.testing.assert_array_equal(pred[1],expectedData2)

    def test_tostring_numbers_wo_dft(self):
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
        
        # testing
        sess = rt.InferenceSession(onnx_path)
        names = []
        for input in sess.get_inputs():
            names.append(input.name)

        output_names = []
        for output in sess.get_outputs():
            output_names.append(output.name)

        f0 = np.array([4, 4, -1, 9]).astype(np.int32).reshape(4,1)
        f1 = np.array([5, 5, 3.1, -0.23]).astype(np.float32).reshape(4,1)
        f2 = np.array([6, 6.7, float("NaN"), float("NaN")]).astype(np.float64).reshape(4,1)
 
        # Run your inference session with your model and your data
        pred_list = sess.run(None, {"f0":f0, "f1": f1, "f2":f2  })
 
        pred = {output_names[i] : pred_list[i] for i in range(0, len(output_names))}
        
        f0_output = np.array([['4'], ['4'], ['-1'], ['9']]).reshape(4, 1)
        f1_output = np.array([['5.000000'], ['5.000000'], ['3.100000'], ['-0.230000']]).reshape(4, 1)
        f2_output = np.array([['6.000000'], ['6.700000'], ['NaN'], ['NaN']]).reshape(4, 1)
        np.testing.assert_array_equal(f0_output, pred_list[3])
        np.testing.assert_array_equal(f1_output, pred_list[4])
        np.testing.assert_array_equal(f2_output, pred_list[5])



    def test_tostring_other_types_wo_dft(self):
        data={'f0': [True, False],
              'f1': [123.45, 135453984983490.5473]}
        data = pd.DataFrame(data).astype({'f0': bool,
                                       'f1': np.double})

        xf = ToString(columns={'f0.out': 'f0',
                               'f1.out': 'f1'})
        xf.fit(data)
        
        onnx_path = get_tmp_file('.onnx')
        onnx_json_path = get_tmp_file('.onnx.json')
        xf.export_to_onnx(onnx_path,
                            'com.microsoft.ml',
                            dst_json=onnx_json_path,
                            onnx_version='Stable')
        
        # testing
        sess = rt.InferenceSession(onnx_path)
        names = []
        for input in sess.get_inputs():
            names.append(input.name)

        output_names = []
        for output in sess.get_outputs():
            output_names.append(output.name)

        f0 = np.array([True, False]).astype(bool).reshape(2,1)
        f1 = np.array([123.45, 135453984983490.5473]).astype(np.double).reshape(2,1)
 
        # Run your inference session with your model and your data
        pred_list = sess.run(None, {"f0":f0, "f1": f1})
 
        pred = {output_names[i] : pred_list[i] for i in range(0, len(output_names))}
        
        f0_output = np.array([['True'], ['False']]).reshape(2, 1)
        f1_output = np.array([['123.450000'], ['135453984983490.546875']]).reshape(2, 1)
        np.testing.assert_array_equal(f0_output, pred_list[2])
        np.testing.assert_array_equal(f1_output, pred_list[3])

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
        xf.fit(df)
        
        onnx_path = get_tmp_file('.onnx')
        onnx_json_path = get_tmp_file('.onnx.json')
        xf.export_to_onnx(onnx_path,
                            'com.microsoft.ml',
                            dst_json=onnx_json_path,
                            onnx_version='Stable')
             
        # testing
        sess = rt.InferenceSession(onnx_path)
        names = []
        for input in sess.get_inputs():
            names.append(input.name)

        output_names = []
        for output in sess.get_outputs():
            output_names.append(output.name)

        ts = np.array([1, 2, 3, 5, 7]).astype(np.int64).reshape(5,1)
        grain = np.array([1970, 1970, 1970, 1970, 1970]).astype(np.int32).reshape(5,1)
        c = np.array([10, 11, 12, 13, 14]).astype(np.int32).reshape(5,1)
 
        # Run your inference session with your model and your data
        pred_list = sess.run(None, {"ts":ts, "grain": grain, 'c':c})
 
        pred = {output_names[i] : pred_list[i] for i in range(0, len(output_names))}
        expected_ts = np.array([[1], [2], [3], [4], [5], [6], [7]]).astype(np.single).reshape(7, 1)
        expected_c = np.array([[10], [11], [12], [12], [13], [13], [14]]).astype(np.single).reshape(7, 1)
        expected_isrowimputed = np.array([[False], [False], [False], [True], [False], [True], [False]]).astype(np.single).reshape(7, 1)
        
        np.testing.assert_array_equal(expected_ts, pred_list[0])
        np.testing.assert_array_equal(expected_c, pred_list[2])
        np.testing.assert_array_equal(expected_isrowimputed, pred_list[3])

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
        xf.fit(df)
        
        onnx_path = get_tmp_file('.onnx')
        onnx_json_path = get_tmp_file('.onnx.json')
        xf.export_to_onnx(onnx_path,
                            'com.microsoft.ml',
                            dst_json=onnx_json_path,
                            onnx_version='Stable')
             
        # testing
        sess = rt.InferenceSession(onnx_path)
        names = []
        for input in sess.get_inputs():
            names.append(input.name)

        output_names = []
        for output in sess.get_outputs():
            output_names.append(output.name)

        ts = np.array([1, 5, 2, 6]).astype(np.int64).reshape(4,1)
        grain = np.array([1970, 1971, 1970, 1971]).astype(np.int32).reshape(4,1)
        c = np.array([10, 11, 12, 13]).astype(np.int32).reshape(4,1)
 
        # Run your inference session with your model and your data
        pred_list = sess.run(None, {"ts":ts, "grain": grain, 'c':c})
 
        pred = {output_names[i] : pred_list[i] for i in range(0, len(output_names))}
        expected_ts = np.array([[1], [5], [2], [6]]).astype(np.single).reshape(4, 1)
        expected_grain = np.array([[1970], [1971], [1970], [1971]]).astype(np.single).reshape(4, 1)
        expected_c = np.array([[10], [11], [12], [13]]).astype(np.single).reshape(4, 1)
        expected_isrowimputed = np.array([[False], [False], [False], [False]]).astype(np.single).reshape(4, 1)
        

        np.testing.assert_array_equal(expected_ts, pred_list[0])
        np.testing.assert_array_equal(expected_grain, pred_list[1])
        np.testing.assert_array_equal(expected_c, pred_list[2])
        np.testing.assert_array_equal(expected_isrowimputed, pred_list[3])

    def test_timeseriesimputer_twograin_twogap(self):

        df = pd.DataFrame(data=dict(
            ts=[0, 5, 1, 6],
            grain=[1970, 1971, 1970, 1971],
            c=[10, 11, 12, 13]
        )).astype({'ts': np.int64, 'grain': np.int32, 'c': np.int32})


        xf = TimeSeriesImputer(time_series_column='ts',
                                filter_columns=['c'],
                                grain_columns=['grain'],
                                impute_mode='ForwardFill',
                                filter_mode='Include')
        xf.fit(df)
        
        onnx_path = get_tmp_file('.onnx')
        onnx_json_path = get_tmp_file('.onnx.json')
        xf.export_to_onnx(onnx_path,
                            'com.microsoft.ml',
                            dst_json=onnx_json_path,
                            onnx_version='Stable')
             
        # testing
        sess = rt.InferenceSession(onnx_path)
        names = []
        for input in sess.get_inputs():
            names.append(input.name)

        output_names = []
        for output in sess.get_outputs():
            output_names.append(output.name)

        ts = np.array([0, 5, 3, 8]).astype(np.int64).reshape(4,1)
        grain = np.array([1970, 1971, 1970, 1971]).astype(np.int32).reshape(4,1)
        c = np.array([10, 11, 12, 13]).astype(np.int32).reshape(4,1)
 
        # Run your inference session with your model and your data
        pred_list = sess.run(None, {"ts":ts, "grain": grain, 'c':c})
 
        pred = {output_names[i] : pred_list[i] for i in range(0, len(output_names))}
        expected_ts = np.array([[0], [5], [1], [2], [3], [6], [7], [8]]).astype(np.single).reshape(8, 1)
        expected_grain = np.array([[1970], [1971], [1970], [1970], [1970], [1971], [1971], [1971]]).astype(np.single).reshape(8, 1)
        expected_c = np.array([[10], [11], [10], [10], [12], [11], [11], [13]]).astype(np.single).reshape(8, 1)
        expected_isrowimputed = np.array([[False], [False], [True], [True], [False], [True], [True], [False]]).astype(np.single).reshape(8, 1)
 
        np.testing.assert_array_equal(expected_ts, pred_list[0])
        np.testing.assert_array_equal(expected_grain, pred_list[1])
        np.testing.assert_array_equal(expected_c, pred_list[2])
        np.testing.assert_array_equal(expected_isrowimputed, pred_list[3])


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
        xf.fit(df)
        
        onnx_path = get_tmp_file('.onnx')
        onnx_json_path = get_tmp_file('.onnx.json')
        xf.export_to_onnx(onnx_path,
                            'com.microsoft.ml',
                            dst_json=onnx_json_path,
                            onnx_version='Stable')
             
        # testing
        sess = rt.InferenceSession(onnx_path)
        names = []
        for input in sess.get_inputs():
            names.append(input.name)

        output_names = []
        for output in sess.get_outputs():
            output_names.append(output.name)

        ts = np.array([1, 2, 3, 5]).astype(np.int64).reshape(4,1)
        grain = np.array([1970, 1970, 1970, 1970]).astype(np.int32).reshape(4,1)
        c3 = np.array([10, 13, 15, 20]).astype(np.int32).reshape(4,1)
        c4 = np.array([19, 12, 16, 19]).astype(np.int32).reshape(4,1)
 
        # Run your inference session with your model and your data
        pred_list = sess.run(None, {"ts":ts, "grain": grain, "c3": c3, "c4": c4  })
 
        pred = {output_names[i] : pred_list[i] for i in range(0, len(output_names))}
        expected_ts = np.array([[1], [2], [3], [4], [5]]).astype(np.single).reshape(5, 1)
        expected_c3 = np.array([[10], [13], [15], [15], [20]]).astype(np.single).reshape(5, 1)
        expected_c4 = np.array([[19], [12], [16], [16], [19]]).astype(np.single).reshape(5, 1)
        expected_isrowimputed = np.array([[False], [False], [False], [True], [False]]).astype(np.single).reshape(5, 1)
        
        np.testing.assert_array_equal(expected_ts, pred_list[0])
        np.testing.assert_array_equal(expected_c3, pred_list[2])
        np.testing.assert_array_equal(expected_c4, pred_list[3])
        np.testing.assert_array_equal(expected_isrowimputed, pred_list[4])

    def test_rolling_window_simple_mean_wo_dft(self):

        df = pd.DataFrame(data=dict(
            colA=[1.0, 2.0, 3.0, 4.0],
            grainA=["one", "one", "one", "one"]
        ))

        xf = RollingWindow(columns={'colA1': 'colA'},
                           grain_column=['grainA'], 
                           window_calculation='Mean',
                           max_window_size=2,
                           horizon=2)
        xf.fit(df)

        onnx_path = get_tmp_file('.onnx')
        onnx_json_path = get_tmp_file('.onnx.json')
        xf.export_to_onnx(onnx_path,
                            'com.microsoft.ml',
                            dst_json=onnx_json_path,
                            onnx_version='Stable')

        # testing
        sess = rt.InferenceSession(onnx_path)
        names = []
        for input in sess.get_inputs():
            names.append(input.name)

        output_names = []
        for output in sess.get_outputs():
            output_names.append(output.name)

        grainA = np.array(["one", "one", "one", "one"]).reshape(4,1)
        colA = np.array([1.0, 2.0, 3.0, 4.0]).astype(np.double).reshape(4,1)
 
        # Run your inference session with your model and your data
        pred_list = sess.run(None, {"grainA":grainA, "colA": colA  })
        
        pred = {output_names[i] : pred_list[i] for i in range(0, len(output_names))}
        
        expected = np.array([[[float("NaN"), float("NaN")]], [[float("NaN"), 1.0]], [[1.0, 1.5]], [[1.5, 2.5]]]).astype(np.single).reshape(4, 1, 2)
        
        np.testing.assert_array_equal(expected, pred_list[2])

    def test_rolling_window_simple_max_wo_dft(self):

        df = pd.DataFrame(data=dict(
            colA=[1.0, 2.0, 3.0, 4.0],
            grainA=["one", "one", "one", "one"]
        ))

        xf = RollingWindow(columns={'colA1': 'colA'},
                           grain_column=['grainA'], 
                           window_calculation='Max',
                           max_window_size=1,
                           horizon=1)
        xf.fit(df)
        
        onnx_path = get_tmp_file('.onnx')
        onnx_json_path = get_tmp_file('.onnx.json')
        xf.export_to_onnx(onnx_path,
                            'com.microsoft.ml',
                            dst_json=onnx_json_path,
                            onnx_version='Stable')

        # testing
        sess = rt.InferenceSession(onnx_path)
        names = []
        for input in sess.get_inputs():
            names.append(input.name)

        output_names = []
        for output in sess.get_outputs():
            output_names.append(output.name)

        grainA = np.array(["one", "one", "one", "one"]).reshape(4,1)
        colA = np.array([1.0, 2.0, 3.0, 4.0]).astype(np.double).reshape(4,1)
 
        # Run your inference session with your model and your data
        pred_list = sess.run(None, {"grainA":grainA, "colA": colA  })
        pred = {output_names[i] : pred_list[i] for i in range(0, len(output_names))}
        
        expected = np.array([[[float("NaN")]], [[1.0]], [[2.0]], [[3.0]]]).astype(np.single).reshape(4, 1, 1)
        
        np.testing.assert_array_equal(expected, pred_list[2])

    def test_rolling_window_simple_min_wo_dft(self):

        df = pd.DataFrame(data=dict(
            colA=[1.0, 2.0, 3.0, 4.0],
            grainA=["one", "one", "one", "one"]
        ))

        xf = RollingWindow(columns={'colA1': 'colA'},
                           grain_column=['grainA'], 
                           window_calculation='Min',
                           max_window_size=1,
                           horizon=1)
        xf.fit(df)
        
        onnx_path = get_tmp_file('.onnx')
        onnx_json_path = get_tmp_file('.onnx.json')
        xf.export_to_onnx(onnx_path,
                            'com.microsoft.ml',
                            dst_json=onnx_json_path,
                            onnx_version='Stable')

        # testing
        sess = rt.InferenceSession(onnx_path)
        names = []
        for input in sess.get_inputs():
            names.append(input.name)

        output_names = []
        for output in sess.get_outputs():
            output_names.append(output.name)

        grainA = np.array(["one", "one", "one", "one"]).reshape(4,1)
        colA = np.array([1.0, 2.0, 3.0, 4.0]).astype(np.double).reshape(4,1)
 
        # Run your inference session with your model and your data
        pred_list = sess.run(None, {"grainA":grainA, "colA": colA  })
        
        pred = {output_names[i] : pred_list[i] for i in range(0, len(output_names))}
        
        expected = np.array([[[float("NaN")]], [[1.0]], [[2.0]], [[3.0]]]).astype(np.single).reshape(4, 1, 1)
        
        np.testing.assert_array_equal(expected, pred_list[2])

    def test_rolling_window_multiple_grains_wo_dft(self):

        df = pd.DataFrame(data=dict(
            colA=[1.0, 2.0, 1.0, 2.0],
            grainA=["one", "one", "two", "two"]
        ))

        xf = RollingWindow(columns={'colA1': 'colA'},
                           grain_column=['grainA'], 
                           window_calculation='Mean',
                           max_window_size=1,
                           horizon=1)
        xf.fit(df)
        
        onnx_path = get_tmp_file('.onnx')
        onnx_json_path = get_tmp_file('.onnx.json')
        xf.export_to_onnx(onnx_path,
                            'com.microsoft.ml',
                            dst_json=onnx_json_path,
                            onnx_version='Stable')

        # testing
        sess = rt.InferenceSession(onnx_path)
        names = []
        for input in sess.get_inputs():
            names.append(input.name)

        output_names = []
        for output in sess.get_outputs():
            output_names.append(output.name)

        grainA = np.array(["one", "one", "two", "two"]).reshape(4,1)
        colA = np.array([1.0, 2.0, 1.0, 2.0]).astype(np.double).reshape(4,1)
 
        # Run your inference session with your model and your data
        pred_list = sess.run(None, {"grainA":grainA, "colA": colA  })
        
        pred = {output_names[i] : pred_list[i] for i in range(0, len(output_names))}
        
        expected = np.array([[[float("NaN")]], [[1.0]], [[float("NaN")]], [[1.0]]]).astype(np.single).reshape(4, 1, 1)
        
        np.testing.assert_array_equal(expected, pred_list[2])
    
    def test_rolling_window_non_string_grain_wo_dft(self):

        df = pd.DataFrame(data=dict(
            colA=[1.0, 2.0, 3.0, 4.0],
            grainA=[True, True, True, True]
        ))

        xf = RollingWindow(columns={'colA1': 'colA'},
                           grain_column=['grainA'], 
                           window_calculation='Mean',
                           max_window_size=1,
                           horizon=1)
        xf.fit(df)
        
        onnx_path = get_tmp_file('.onnx')
        onnx_json_path = get_tmp_file('.onnx.json')
        xf.export_to_onnx(onnx_path,
                            'com.microsoft.ml',
                            dst_json=onnx_json_path,
                            onnx_version='Stable')

        # testing
        sess = rt.InferenceSession(onnx_path)
        names = []
        for input in sess.get_inputs():
            names.append(input.name)

        output_names = []
        for output in sess.get_outputs():
            output_names.append(output.name)

        grainA = np.array([True, True, True, True]).reshape(4,1)
        colA = np.array([1.0, 2.0, 3.0, 4.0]).astype(np.double).reshape(4,1)
 
        # Run your inference session with your model and your data
        pred_list = sess.run(None, {"grainA":grainA, "colA": colA  })
        
        pred = {output_names[i] : pred_list[i] for i in range(0, len(output_names))}
        
        expected = np.array([[[float("NaN")]], [[1.0]], [[2.0]], [[3.0]]]).astype(np.single).reshape(4, 1, 1)
        
        np.testing.assert_array_equal(expected, pred_list[2])
    
    def test_laglead_simplelag_wo_dft(self):

        df = pd.DataFrame(data=dict(
            colA=[1.0, 2.0, 3.0, 4.0],
            grainA=["one", "one", "one", "one"]
        ))

        xf = LagLeadOperator(columns={'colA1': 'colA'},
                           grain_columns=['grainA'], 
                           offsets=[-1],
                           horizon=1)
        xf.fit(df)

        onnx_path = get_tmp_file('.onnx')
        onnx_json_path = get_tmp_file('.onnx.json')
        xf.export_to_onnx(onnx_path,
                            'com.microsoft.ml',
                            dst_json=onnx_json_path,
                            onnx_version='Stable')
        
        # testing
        sess = rt.InferenceSession(onnx_path)
        names = []
        for input in sess.get_inputs():
            names.append(input.name)

        output_names = []
        for output in sess.get_outputs():
            output_names.append(output.name)

        grainA = np.array(["one", "one", "one", "one"]).reshape(4,1)
        colA = np.array([1, 2, 3, 4]).astype(np.double).reshape(4,1)
 
        # Run your inference session with your model and your data
        pred_list = sess.run(None, {"grainA":grainA, "colA": colA  })
 
        pred = {output_names[i] : pred_list[i] for i in range(0, len(output_names))}
        
        expected = np.array([[[float("NaN")]], [[1.0]], [[2.0]], [[3.0]]]).astype(np.single).reshape(4, 1, 1)
        
        np.testing.assert_array_equal(expected, pred_list[2])

    def test_laglead_lag_wo_dft(self):

        df = pd.DataFrame(data=dict(
            colA=[1.0, 2.0, 3.0, 4.0],
            grainA=["one", "one", "one", "one"]
        ))

        xf = LagLeadOperator(columns={'colA1': 'colA'},
                           grain_columns=['grainA'], 
                           offsets=[-2, -1],
                           horizon=2)
        xf.fit(df)

        onnx_path = get_tmp_file('.onnx')
        onnx_json_path = get_tmp_file('.onnx.json')
        xf.export_to_onnx(onnx_path,
                            'com.microsoft.ml',
                            dst_json=onnx_json_path,
                            onnx_version='Stable')
        
        # testing
        sess = rt.InferenceSession(onnx_path)
        names = []
        for input in sess.get_inputs():
            names.append(input.name)

        output_names = []
        for output in sess.get_outputs():
            output_names.append(output.name)

        grainA = np.array(["one", "one", "one", "one"]).reshape(4,1)
        colA = np.array([1, 2, 3, 4]).astype(np.double).reshape(4,1)
 
        # Run your inference session with your model and your data
        pred_list = sess.run(None, {"grainA":grainA, "colA": colA  })
 
        pred = {output_names[i] : pred_list[i] for i in range(0, len(output_names))}
        
        
        expected = np.array([[[float("NaN"), float("NaN")], [float("NaN"), float("NaN")]], [[float("NaN"), float("NaN")], [float("NaN"), 1.0]], [[float("NaN"), 1.0], [1.0, 2.0]], [[1.0, 2.0], [2.0, 3.0]]]).astype(np.single).reshape(4, 2, 2)
        
        np.testing.assert_array_equal(expected, pred_list[2])

    def test_laglead_lead_wo_dft(self):

        df = pd.DataFrame(data=dict(
            colA=[1.0, 2.0, 3.0, 4.0],
            grainA=["one", "one", "one", "one"]
        ))

        xf = LagLeadOperator(columns={'colA1': 'colA'},
                           grain_columns=['grainA'], 
                           offsets=[1, 2],
                           horizon=2)
        xf.fit(df)

        onnx_path = get_tmp_file('.onnx')
        onnx_json_path = get_tmp_file('.onnx.json')
        xf.export_to_onnx(onnx_path,
                            'com.microsoft.ml',
                            dst_json=onnx_json_path,
                            onnx_version='Stable')
        
        # testing
        sess = rt.InferenceSession(onnx_path)
        names = []
        for input in sess.get_inputs():
            names.append(input.name)

        output_names = []
        for output in sess.get_outputs():
            output_names.append(output.name)

        grainA = np.array(["one", "one", "one", "one"]).reshape(4,1)
        colA = np.array([1, 2, 3, 4]).astype(np.double).reshape(4,1)
 
        # Run your inference session with your model and your data
        pred_list = sess.run(None, {"grainA":grainA, "colA": colA  })
 
        pred = {output_names[i] : pred_list[i] for i in range(0, len(output_names))}
        
        
        expected = np.array([[[1.0, 2.0], [2.0, 3.0]], [[2.0, 3.0], [3.0, 4.0]], [[3.0, 4.0], [4.0, float("NaN")]], [[4.0, float("NaN")], [float("NaN"), float("NaN")]]]).astype(np.single).reshape(4, 2, 2)
        
        np.testing.assert_array_equal(expected, pred_list[2])
    
    def test_laglead_complex_wo_dft(self):

        df = pd.DataFrame(data=dict(
            colA=[1.0, 2.0, 3.0, 4.0],
            grainA=["one", "one", "one", "one"]
        ))

        xf = LagLeadOperator(columns={'colA1': 'colA'},
                           grain_columns=['grainA'], 
                           offsets=[-2, -1, 1, 2],
                           horizon=2)
        xf.fit(df)

        onnx_path = get_tmp_file('.onnx')
        onnx_json_path = get_tmp_file('.onnx.json')
        xf.export_to_onnx(onnx_path,
                            'com.microsoft.ml',
                            dst_json=onnx_json_path,
                            onnx_version='Stable')
        
        # testing
        sess = rt.InferenceSession(onnx_path)
        names = []
        for input in sess.get_inputs():
            names.append(input.name)

        output_names = []
        for output in sess.get_outputs():
            output_names.append(output.name)

        grainA = np.array(["one", "one", "one", "one"]).reshape(4,1)
        colA = np.array([1, 2, 3, 4]).astype(np.double).reshape(4,1)
 
        # Run your inference session with your model and your data
        pred_list = sess.run(None, {"grainA":grainA, "colA": colA  })
 
        pred = {output_names[i] : pred_list[i] for i in range(0, len(output_names))}
        
        
        expected = np.array([[[float("NaN"), float("NaN")], [float("NaN"), float("NaN")], [1.0, 2.0], [2.0, 3.0]], 
                            [[float("NaN"), float("NaN")], [float("NaN"), 1.0], [2.0, 3.0], [3.0, 4.0]], 
                            [[float("NaN"), 1.0], [1.0, 2.0], [3.0, 4.0], [4.0, float("NaN")]], 
                            [[1.0, 2.0], [2.0, 3.0], [4.0, float("NaN")], [float("NaN"), float("NaN")]]]).astype(np.single).reshape(4, 4, 2)
        
        np.testing.assert_array_equal(expected, pred_list[2])
        
    def test_short_drop_wo_dft(self):

        df = pd.DataFrame(data=dict(
            colA=[1.0, 2.0, 3.0, 4.0],
            grainA=["one", "one", "one", "two"]
        ))

        xf = ShortDrop(columns={'colA1': 'colA'},
                           grain_columns=['grainA'],
                           min_rows=2)
        xf.fit(df)

        onnx_path = get_tmp_file('.onnx')
        onnx_json_path = get_tmp_file('.onnx.json')
        xf.export_to_onnx(onnx_path,
                            'com.microsoft.ml',
                            dst_json=onnx_json_path,
                            onnx_version='Stable')

        # testing
        sess = rt.InferenceSession(onnx_path)
        names = []
        for input in sess.get_inputs():
            names.append(input.name)

        output_names = []
        for output in sess.get_outputs():
            output_names.append(output.name)

        grainA = np.array(["one", "one", "one", "two"]).reshape(4,1)
        colA = np.array([1, 2, 3, 4]).astype(np.double).reshape(4,1)
 
        # Run your inference session with your model and your data
        pred_list = sess.run(None, {"grainA":grainA, "colA": colA  })
 
        pred = {output_names[i] : pred_list[i] for i in range(0, len(output_names))}
        
        
        colA_expected = np.array([[1.0], [2.0], [3.0]]).reshape(3, 1)
        grainA_expected = np.array([["one"], ["one"], ["one"]]).reshape(3, 1)
        np.testing.assert_array_equal(colA_expected, pred_list[0])
        np.testing.assert_array_equal(grainA_expected, pred_list[1])



    def test_integration_rollwin_pivot(self):

        df = pd.DataFrame(data=dict(
            colA=[1.0, 3.0, 5.0, 7.0],
            grainA=['1970', '1970', '1970', '1970']
        ))

        xf0 = RollingWindow(columns={'colA1': 'colA'},
                           grain_column=['grainA'], 
                           window_calculation='Mean',
                           max_window_size=2,
                           horizon=2)

        xf1 = ForecastingPivot(columns_to_pivot=['colA1'])

        xf = Pipeline([xf0, xf1])

        xf.fit(df)
        
        onnx_path = get_tmp_file('.onnx')
        onnx_json_path = get_tmp_file('.onnx.json')
        xf.export_to_onnx(onnx_path,
                            'com.microsoft.ml',
                            dst_json=onnx_json_path,
                            onnx_version='Stable')
                            
        # testing
        sess = rt.InferenceSession(onnx_path)
        names = []
        for input in sess.get_inputs():
            names.append(input.name)

        output_names = []
        for output in sess.get_outputs():
            output_names.append(output.name)

        colA = np.array([1.0, 3.0, 5.0, 7.0]).reshape(4,1)
        grainA = np.array(['1970', '1970', '1970', '1970']).reshape(4,1)
 
        # Run your inference session with your model and your data
        pred_list = sess.run(None, {"colA":colA, "grainA": grainA  })
 
        pred = {output_names[i] : pred_list[i] for i in range(0, len(output_names))}
        
        expected_colA = np.array([[3.0], [5.0], [5.0], [7.0], [7.0]])
        expected_grainA = np.array([['1970'], ['1970'], ['1970'], ['1970'], ['1970']])
        expected_output = np.array([[1.0], [1.0], [2.0], [2.0], [4.0]])
        np.testing.assert_array_equal(expected_colA, pred_list[0])
        np.testing.assert_array_equal(expected_grainA, pred_list[1])
        np.testing.assert_array_equal(expected_output, pred_list[3])
        


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

        xf.fit(df)
        
        onnx_path = get_tmp_file('.onnx')
        onnx_json_path = get_tmp_file('.onnx.json')
        xf.export_to_onnx(onnx_path,
                            'com.microsoft.ml',
                            dst_json=onnx_json_path,
                            onnx_version='Stable')
                            
        # testing
        sess = rt.InferenceSession(onnx_path)
        names = []
        for input in sess.get_inputs():
            names.append(input.name)

        output_names = []
        for output in sess.get_outputs():
            output_names.append(output.name)

        grainA = np.array(["one", "one", "one", "one"]).reshape(4,1)
        colA = np.array([1, 2, 3, 4]).astype(np.double).reshape(4,1)
 
        # Run your inference session with your model and your data
        pred_list = sess.run(None, {"colA":colA, "grainA": grainA  })
 
        pred = {output_names[i] : pred_list[i] for i in range(0, len(output_names))}
        
        expected_colA = np.array([[3.0], [4.0], [4.0]])
        expected_grainA = np.array([['one'], ['one'], ['one']])
        expected_output_lag2 = np.array([[1.0], [1.0], [2.0]])
        expected_output_lag1 = np.array([[2.0], [2.0], [3.0]])
        np.testing.assert_array_equal(expected_colA, pred_list[0])
        np.testing.assert_array_equal(expected_grainA, pred_list[1])
        np.testing.assert_array_equal(expected_output_lag2, pred_list[3])
        np.testing.assert_array_equal(expected_output_lag1, pred_list[4])


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

if __name__ == '__main__':
    unittest.main()
