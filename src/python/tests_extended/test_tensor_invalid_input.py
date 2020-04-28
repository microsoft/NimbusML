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

TEST_CASES_FOR_INVALID_INPUT = {
    'DateTimeSplitter_Bad_Input_Data',
    'DateTimeSplitter_Bad_Input_Type',
    'DateTimeSplitter_Bad_Input_Shape',
    'ToKey_Bad_Input_Type',
    'ToKey_Bad_Input_Shape',
    'ToString_Bad_Input_Type',
    'ToString_Bad_Input_Shape',
    'TimeSeriesImputer_Bad_Input_Data',
    'TimeSeriesImputer_Bad_Input_Type',
    'TimeSeriesImputer_Bad_Input_Shape',
    'RollingWindow_Bad_Input_Type',
    'RollingWindow_Bad_Input_Shape',
    'LagLead_Bad_Input_Type',
    'LagLead_Bad_Input_Shape',
    'ShortDrop_Bad_Input_Type',
    'ShortDrop_Bad_Input_Shape',
    'ShortDrop_Drop_All'
}

INSTANCES_FOR_INVALID_INPUT = {
    'DateTimeSplitter_Bad_Input_Data': Pipeline([
        DateTimeSplitter(prefix='dt') << 'tokens1',
        ColumnSelector(drop_columns=[
                                    'dtHour12', 'dtDayOfWeek', 'dtDayOfQuarter',
                                    'dtDayOfYear', 'dtWeekOfMonth', 'dtQuarterOfYear',
                                    'dtHalfOfYear', 'dtWeekIso', 'dtYearIso', 'dtMonthLabel',
                                    'dtAmPmLabel', 'dtDayOfWeekLabel', 'dtIsPaidTimeOff','dtHolidayName'
        ])
    ]),
    'DateTimeSplitter_Bad_Input_Type': Pipeline([
        DateTimeSplitter(prefix='dt') << 'tokens1',
        ColumnSelector(drop_columns=[
                                    'dtHour12', 'dtDayOfWeek', 'dtDayOfQuarter',
                                    'dtDayOfYear', 'dtWeekOfMonth', 'dtQuarterOfYear',
                                    'dtHalfOfYear', 'dtWeekIso', 'dtYearIso', 'dtMonthLabel',
                                    'dtAmPmLabel', 'dtDayOfWeekLabel', 'dtIsPaidTimeOff','dtHolidayName'
        ])
    ]),
    'DateTimeSplitter_Bad_Input_Shape': Pipeline([
        DateTimeSplitter(prefix='dt') << 'tokens1',
        ColumnSelector(drop_columns=[
                                    'dtHour12', 'dtDayOfWeek', 'dtDayOfQuarter',
                                    'dtDayOfYear', 'dtWeekOfMonth', 'dtQuarterOfYear',
                                    'dtHalfOfYear', 'dtWeekIso', 'dtYearIso', 'dtMonthLabel',
                                    'dtAmPmLabel', 'dtDayOfWeekLabel', 'dtIsPaidTimeOff','dtHolidayName'
        ])
    ]),
    'ToKey_Bad_Input_Type': ToKeyImputer(),
    'ToKey_Bad_Input_Shape': ToKeyImputer(),
    'ToString_Bad_Input_Type': ToString(columns={'f0.out': 'f0',
                                                'f1.out': 'f1',
                                                'f2.out': 'f2'}),
    'ToString_Bad_Input_Shape': ToString(columns={'f0.out': 'f0',
                                                'f1.out': 'f1',
                                                'f2.out': 'f2'}),
    'TimeSeriesImputer_Bad_Input_Data': TimeSeriesImputer(time_series_column='ts',
                                                          filter_columns=['c'],
                                                          grain_columns=['grain'],
                                                          impute_mode='ForwardFill',
                                                          filter_mode='Include'),
    'TimeSeriesImputer_Bad_Input_Type': TimeSeriesImputer(time_series_column='ts',
                                                          filter_columns=['c'],
                                                          grain_columns=['grain'],
                                                          impute_mode='ForwardFill',
                                                          filter_mode='Include'),
    'TimeSeriesImputer_Bad_Input_Shape': TimeSeriesImputer(time_series_column='ts',
                                                          filter_columns=['c'],
                                                          grain_columns=['grain'],
                                                          impute_mode='ForwardFill',
                                                          filter_mode='Include'),
    'RollingWindow_Bad_Input_Type': RollingWindow(columns={'colA1': 'colA'},
                                                  grain_columns=['grainA'], 
                                                  window_calculation='Mean',
                                                  max_window_size=2,
                                                  horizon=2),
    'RollingWindow_Bad_Input_Shape': RollingWindow(columns={'colA1': 'colA'},
                                                  grain_columns=['grainA'], 
                                                  window_calculation='Mean',
                                                  max_window_size=2,
                                                  horizon=2),
    'LagLead_Bad_Input_Type': LagLeadOperator(columns={'colA1': 'colA'},
                                              grain_columns=['grainA'], 
                                              offsets=[-1],
                                              horizon=1),
    'LagLead_Bad_Input_Shape': LagLeadOperator(columns={'colA1': 'colA'},
                                              grain_columns=['grainA'], 
                                              offsets=[-1],
                                              horizon=1),
    'ShortDrop_Bad_Input_Type': ShortDrop(columns={'colA1': 'colA'},
                                           grain_columns=['grainA'],
                                           min_rows=2),
    'ShortDrop_Bad_Input_Shape': ShortDrop(columns={'colA1': 'colA'},
                                           grain_columns=['grainA'],
                                           min_rows=2),
    'ShortDrop_Drop_All': ShortDrop(columns={'colA1': 'colA'},
                                           grain_columns=['grainA'],
                                           min_rows=15)
}

TRAINING_DATASETS_FOR_INVALID_INPUT = {
    'DateTimeSplitter_Bad_Input_Data': pd.DataFrame(data=dict(
                                           tokens1=[1, 2, 3, 157161600]
                                       )),
    'DateTimeSplitter_Bad_Input_Type': pd.DataFrame(data=dict(
                                           tokens1=[1, 2, 3, 157161600]
                                       )),
    'DateTimeSplitter_Bad_Input_Shape': pd.DataFrame(data=dict(
                                           tokens1=[1, 2, 3, 157161600]
                                       )),
    'ToKey_Bad_Input_Type': pd.DataFrame(data=dict(
                                target=[1.0, 1.0, 1.0, 2.0]
                            )).astype({'target': np.float64}),
    'ToKey_Bad_Input_Shape': pd.DataFrame(data=dict(
                                target=[1.0, 1.0, 1.0, 2.0]
                            )).astype({'target': np.float64}),
    'ToString_Bad_Input_Type': pd.DataFrame(data=dict(
                                            f0= [4, 4, -1, 9],
                                            f1= [5, 5, 3.1, -0.23],
                                            f2= [6, 6.7, np.nan, np.nan]
                               )),
    'ToString_Bad_Input_Shape': pd.DataFrame(data=dict(
                                            f0= [4, 4, -1, 9],
                                            f1= [5, 5, 3.1, -0.23],
                                            f2= [6, 6.7, np.nan, np.nan]
                                )),
    'TimeSeriesImputer_Bad_Input_Data': pd.DataFrame(data=dict(
                                            ts=[1, 2, 3, 5, 7],
                                            grain=[1970, 1970, 1970, 1970, 1970],
                                            c=[10, 11, 12, 13, 14]
                                        )).astype({'ts': np.int64, 'grain': np.int32, 'c': np.int32}),
    'TimeSeriesImputer_Bad_Input_Type': pd.DataFrame(data=dict(
                                            ts=[1, 2, 3, 5, 7],
                                            grain=[1970, 1970, 1970, 1970, 1970],
                                            c=[10, 11, 12, 13, 14]
                                        )).astype({'ts': np.int64, 'grain': np.int32, 'c': np.int32}),
    'TimeSeriesImputer_Bad_Input_Shape': pd.DataFrame(data=dict(
                                            ts=[1, 2, 3, 5, 7],
                                            grain=[1970, 1970, 1970, 1970, 1970],
                                            c=[10, 11, 12, 13, 14]
                                        )).astype({'ts': np.int64, 'grain': np.int32, 'c': np.int32}),
    'RollingWindow_Bad_Input_Type': pd.DataFrame(data=dict(
                                        colA=[1.0, 2.0, 3.0, 4.0],
                                        grainA=["one", "one", "one", "one"]
                                    )),
    'RollingWindow_Bad_Input_Shape': pd.DataFrame(data=dict(
                                        colA=[1.0, 2.0, 3.0, 4.0],
                                        grainA=["one", "one", "one", "one"]
                                    )),
    'LagLead_Bad_Input_Type':pd.DataFrame(data=dict(
                                 colA=[1.0, 2.0, 3.0, 4.0],
                                 grainA=["one", "one", "one", "one"]
                             )),
    'LagLead_Bad_Input_Shape': pd.DataFrame(data=dict(
                                 colA=[1.0, 2.0, 3.0, 4.0],
                                 grainA=["one", "one", "one", "one"]
                               )),
    'ShortDrop_Bad_Input_Type': pd.DataFrame(data=dict(
                                     colA=[1.0, 2.0, 3.0, 4.0],
                                     grainA=["one", "one", "one", "two"]
                                )),
    'ShortDrop_Bad_Input_Shape': pd.DataFrame(data=dict(
                                     colA=[1.0, 2.0, 3.0, 4.0],
                                     grainA=["one", "one", "one", "two"]
                                 )),
    'ShortDrop_Drop_All': pd.DataFrame(data=dict(
                                     colA=[1.0, 2.0, 3.0, 4.0],
                                     grainA=["one", "one", "one", "two"]
                                 ))
}

INFERENCE_DATASETS_FOR_INVALID_INPUT = {
    'DateTimeSplitter_Bad_Input_Data': {"tokens1": np.array([1, 2, 3, -3]).reshape(4,1)},
    'DateTimeSplitter_Bad_Input_Type': {"tokens1": np.array([1, 2, 3, "3"]).reshape(4,1)},
    'DateTimeSplitter_Bad_Input_Shape': {"tokens1": np.array([[1, 2, 3, 3], [1, 2, 3, 4]]).reshape(4,2)},
    'ToKey_Bad_Input_Type': {"target": np.array([1, float("NaN"), "", 7, float("NaN")]).reshape(5,1)},
    'ToKey_Bad_Input_Shape': {"target": np.array([[1, float("NaN")], [1, float("NaN")]]).reshape(2,2)},
    'ToString_Bad_Input_Type': {"f0": np.array([4, 4, -1, 9]).astype(np.int32).reshape(4,1),
                                "f1": np.array([5, 5, 3.1, -0.23]).astype(np.float32).reshape(4,1),
                                "f2": np.array([6, "6.7", float("NaN"), float("NaN")]).reshape(4,1)},
    'ToString_Bad_Input_Shape': {"f0": np.array([[4, 4, -1, 9],[4, 4, -1, 9]]).astype(np.int32).reshape(4,2),
                                 "f1": np.array([5, 5, 3.1, -0.23]).astype(np.float32).reshape(4,1),
                                 "f2": np.array([6, 6.7, float("NaN"), float("NaN")]).reshape(4,1)},
    'TimeSeriesImputer_Bad_Input_Data': {"ts": np.array([1, 2, 3, -5, 7]).astype(np.int64).reshape(5,1),
                                         "grain": np.array([1970, 1970, 1970, 1970, 1970]).reshape(5,1),
                                         "c": np.array([10, 11, 12, 13, 14]).astype(np.int32).reshape(5,1)},
    'TimeSeriesImputer_Bad_Input_Type': {"ts": np.array([1, 2, 3, 5, 7]).astype(np.int64).reshape(5,1),
                                         "grain": np.array([1970, "1970", 1970, 1970, 1970]).reshape(5,1),
                                         "c": np.array([10, 11, 12, 13, 14]).astype(np.int32).reshape(5,1)},
    'TimeSeriesImputer_Bad_Input_Shape': {"ts": np.array([[1, 2, 3, 5, 7], [1, 2, 3, 5, 7]]).astype(np.int64).reshape(5,2),
                                          "grain": np.array([1970, 1970, 1970, 1970, 1970]).reshape(5,1),
                                          "c": np.array([10, 11, 12, 13, 14]).astype(np.int32).reshape(5,1)},
    'RollingWindow_Bad_Input_Type': {"grainA": np.array(["one", "one", "one", "one"]).reshape(4,1),
                                     "colA": np.array([1.0, 2.0, 3.0, "4.0"]).reshape(4,1)},
    'RollingWindow_Bad_Input_Shape': {"grainA": np.array(["one", "one", "one", "one"]).reshape(4,1),
                                      "colA": np.array([[1.0, 2.0, 3.0, 4.0],[1.0, 2.0, 3.0, 4.0]]).reshape(4,2)},
    'LagLead_Bad_Input_Type': {"grainA": np.array(["one", "one", "one", "one"]).reshape(4,1),
                               "colA": np.array([1, 2, 3, "4"]).reshape(4,1)},
    'LagLead_Bad_Input_Shape': {"grainA": np.array(["one", "one", "one", "one"]).reshape(4,1),
                                "colA": np.array([[1.0, 2.0, 3.0, 4.0],[1.0, 2.0, 3.0, 4.0]]).reshape(4,2)},
    'ShortDrop_Bad_Input_Type': {"grainA": np.array(["one", "one", "one", "two"]).reshape(4,1),
                                 "colA": np.array([1, 2, 3, "4"]).reshape(4,1)},
    'ShortDrop_Bad_Input_Shape': {"grainA": np.array(["one", "one", "one", "two"]).reshape(4,1),
                                  "colA": np.array([[1.0, 2.0, 3.0, 4.0],[1.0, 2.0, 3.0, 4.0]]).reshape(4,2)},
    'ShortDrop_Drop_All': {"grainA": np.array(["one", "one", "one", "two"]).reshape(4,1),
                                  "colA": np.array([[1.0, 2.0, 3.0, 4.0],[1.0, 2.0, 3.0, 4.0]]).reshape(4,2)}
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

def validate_bad_input(self, estimator, test_case):
    onnx_path = get_tmp_file('.onnx')
    onnx_json_path = get_tmp_file('.onnx.json')

    exported = False
    throw_expected_error = False
    try:
        dataset = TRAINING_DATASETS_FOR_INVALID_INPUT.get(test_case)

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

        sess = rt.InferenceSession(onnx_path)

        with self.assertRaisesRegex(Exception, "ONNXRuntimeError"):
            invalid_data = INFERENCE_DATASETS_FOR_INVALID_INPUT.get(test_case)
            pred = sess.run(None, invalid_data)

        throw_expected_error = True

    os.remove(onnx_path)
    os.remove(onnx_json_path)
    return {'exported': exported, 'throw_expected_error': throw_expected_error}

class TestOnnxExport(unittest.TestCase):
    # This method is a static method of the class
    # because there were pytest fixture related
    # issues when the method was in the global scope.
    @staticmethod
    def generate_test_method_for_bad(test_case):
        def method(self):
            estimator = INSTANCES_FOR_INVALID_INPUT[test_case]

            result = validate_bad_input(self, estimator, test_case)
            assert result['exported']
            assert result['throw_expected_error']

        return method
   
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



for test_case_invalid_input in TEST_CASES_FOR_INVALID_INPUT:
    test_name = 'test_%s' % test_case_invalid_input.replace('(', '_').replace(')', '').lower()
    
    method = TestOnnxExport.generate_test_method_for_bad(test_case_invalid_input)
    setattr(TestOnnxExport, test_name, method)
    
if __name__ == '__main__':
    unittest.main()
