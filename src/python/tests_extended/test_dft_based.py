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
    'DateTimeSplitter_Canada_1day_before_christmas',
    'DateTimeSplitter_Czech_non_english_holiday',
    'ToKey_SimpleFloat',
    'ToKey_SimpleDouble',
    'ToKey_SimpleString',
    'ToKey_2col_Double',
    'ToKey_2col_Double_String',
    'ToString_Numbers',
    'ToString_Other_Types',
    'TimeSeriesImputer_1grain_2gap',
    'TimeSeriesImputer_1grain_2gap_backfill',
    'TimeSeriesImputer_1grain_2gap_medianfill',
    'TimeSeriesImputer_1grain_2gap',
    'TimeSeriesImputer_1grain_1gap_2filtercolumn',
    'TimeSeriesImputer_2grain_nogap',
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
                                    'dtAmPmLabel', 'dtDayOfWeekLabel', 'dtIsPaidTimeOff','dtHolidayName'
        ])
    ]),
    'DateTimeSplitter_Complex' : Pipeline([
        DateTimeSplitter(prefix='dt') << 'tokens1',
        ColumnSelector(drop_columns=[
                                    'dtHour12', 'dtDayOfWeek', 'dtDayOfQuarter',
                                    'dtDayOfYear', 'dtWeekOfMonth', 'dtQuarterOfYear',
                                    'dtHalfOfYear', 'dtWeekIso', 'dtYearIso', 'dtMonthLabel',
                                    'dtAmPmLabel', 'dtDayOfWeekLabel', 'dtIsPaidTimeOff','dtHolidayName'
        ])
    ]),
    'DateTimeSplitter_Canada_1day_before_christmas' : DateTimeSplitter(prefix='dt', country='Canada') << 'tokens1',
    'DateTimeSplitter_Czech_non_english_holiday' : DateTimeSplitter(prefix='dt', country='Czech') << 'tokens1',
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
    'TimeSeriesImputer_1grain_2gap_backfill': TimeSeriesImputer(time_series_column='ts',
                                                       filter_columns=['c'],
                                                       grain_columns=['grain'],
                                                       impute_mode='BackFill',
                                                       filter_mode='Include'),
    'TimeSeriesImputer_1grain_2gap_medianfill': TimeSeriesImputer(time_series_column='ts',
                                                       filter_columns=['c'],
                                                       grain_columns=['grain'],
                                                       impute_mode='Median',
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
    'ShortGrainDropper': ShortDrop(columns={'colA1': 'colA'},
                                   grain_columns=['grainA'],
                                   min_rows=2),
    # For RollingWindow with horizon as 2, max_window_size as 2 and we are calculating Mean, assume time frequency is 1 day
    # output for each row would be [[mean of (2DaysBeforeYesterday and DayBeforeYesterday), mean of (DayBeforeYesterday and Yesterday)]]
    # forecasting pivot will spread this 2d vector out and drop rows that have NaNs in it
    'RollingWin_Pivot_Integration': Pipeline([
        RollingWindow(columns={'colA1': 'colA'},
                           grain_column=['grainA'], 
                           window_calculation='Mean',
                           max_window_size=2,
                           horizon=2),
        ForecastingPivot(columns_to_pivot=['colA1'])
    ]),
    # For LagLeadOperator with horizon as 2 and offsets as [-2, -1], assume time frequency is 1 day
    # output for each row would be [[2DaysBeforeYesterday, DayBeforeYesterday],
    #                               [DayBeforeYesterday, Yesterday]]
    # forecasting pivot will spread this 2d vector out and drop rows that have NaNs in it
    'Laglead_Pivot_Integration': Pipeline([
        LagLeadOperator(columns={'colA1': 'colA'},
                        grain_columns=['grainA'], 
                        offsets=[-2, -1],
                        horizon=2),
        ForecastingPivot(columns_to_pivot=['colA1'])
    ]),
}

DATASETS = {
    'DateTimeSplitter_Simple': pd.DataFrame(data=dict(
                                tokens1=[1, 2, 3, 157161600]
                               )),
    'DateTimeSplitter_Complex': pd.DataFrame(data=dict(
                                tokens1=[217081624, 1751241600, 217081625, 32445842582]
                               )),
    'DateTimeSplitter_Canada_1day_before_christmas': pd.DataFrame(data=dict(
                                tokens1=[157161599]
                               )),
    'DateTimeSplitter_Czech_non_english_holiday': pd.DataFrame(data=dict(
                                tokens1=[3911760000, 3834432000, 3985200000]
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
                        )),
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
    'TimeSeriesImputer_1grain_2gap_backfill': pd.DataFrame(data=dict(
                                                  ts=[1, 2, 3, 5, 7],
                                                  grain=[1970, 1970, 1970, 1970, 1970],
                                                  c=[10, 11, 12, 13, 14]
                                              )).astype({'ts': np.int64, 'grain': np.int32, 'c': np.int32}),
    'TimeSeriesImputer_1grain_2gap_medianfill': pd.DataFrame(data=dict(
                                                    ts=[1, 2, 3, 5, 7],
                                                    grain=[1970, 1970, 1970, 1970, 1970],
                                                    c=[10, 11, 12, 13, 14]
                                                )).astype({'ts': np.int64, 'grain': np.int32, 'c': np.double}),
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

def validate_results(result_mlnet, result_ort):
    
    if len(result_ort.columns) != len(result_mlnet.columns):
        raise RuntimeError("ERROR: The ORT output does not contain the same number of columns as ML.NET.")
    col_tuples = list(zip(result_mlnet.columns[0:],
                          result_ort.columns[0:]))
    for col_tuple in col_tuples:
        try:
            col_mlnet = result_mlnet.loc[:, col_tuple[0]]
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
    return True

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
        dataset = DATASETS.get(test_case)
        
        result_mlnet = estimator.fit_transform(dataset)

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
            df_tool = DFT(onnx_path)
            result_ort = df_tool.execute(dataset, [])

            export_valid = validate_results(result_mlnet, result_ort)
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

for test_case in TEST_CASES:
    test_name = 'test_%s' % test_case.replace('(', '_').replace(')', '').lower()
    method = TestOnnxExport.generate_test_method(test_case)
    setattr(TestOnnxExport, test_name, method)

if __name__ == '__main__':
    unittest.main()
