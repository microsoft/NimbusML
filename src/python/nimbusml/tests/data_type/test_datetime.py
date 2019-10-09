# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
import os
import sys
import unittest
import tempfile

import numpy as np
import pandas as pd
from nimbusml import Pipeline, DprepDataStream
from nimbusml.preprocessing.missing_values import Handler


def get_temp_file(suffix=None):
    fd, file_name = tempfile.mkstemp(suffix=suffix)
    fl = os.fdopen(fd, 'w')
    fl.close()
    return file_name


class TestDateTimeDataType(unittest.TestCase):
    def test_negative_values(self):
        milliseconds_in_year = 365*24*60*60*1000
        data = [i * milliseconds_in_year for i in [-1, -2, -3, -3.3]]

        df = pd.DataFrame({'c1': data, 'c2': [3,4,5,6]})
        df = df.astype({'c1': np.dtype('datetime64[ms]')})

        pipeline = Pipeline(steps=[Handler(columns={'c2': 'c2'})])
        result = pipeline.fit_transform(df)

        self.assertTrue(result.loc[:, 'c1'].equals(df.loc[:, 'c1']))
        self.assertEqual(result.loc[:, 'c1'].dtype, np.dtype('datetime64[ns]'))

        self.assertEqual(result.loc[0, 'c1'].year, 1969)
        self.assertEqual(result.loc[0, 'c1'].hour, 0)
        self.assertEqual(result.loc[0, 'c1'].minute, 0)
        self.assertEqual(result.loc[0, 'c1'].second, 0)

        self.assertEqual(result.loc[3, 'c1'].year, 1966)

    def test_timestamp_boundaries(self):
        # Here are the current min and max for a Pandas Timestamp
        # 1677-09-21 00:12:43.145225
        # 2262-04-11 23:47:16.854775807

        data = [pd.Timestamp(1677, 9, 22, 1), pd.Timestamp.max]
        df = pd.DataFrame({'c1': data, 'c2': [3,4]})
        df = df.astype({'c1': np.dtype('datetime64[ms]')})

        pipeline = Pipeline(steps=[Handler(columns={'c2': 'c2'})])
        result = pipeline.fit_transform(df)

        self.assertTrue(result.loc[:, 'c1'].equals(df.loc[:, 'c1']))
        self.assertEqual(result.dtypes[0], np.dtype('datetime64[ns]'))

        self.assertEqual(result.loc[0, 'c1'].year, 1677)
        self.assertEqual(result.loc[0, 'c1'].month, 9)
        self.assertEqual(result.loc[0, 'c1'].day, 22)

        self.assertEqual(result.loc[1, 'c1'].year, 2262)
        self.assertEqual(result.loc[1, 'c1'].month, 4)
        self.assertEqual(result.loc[1, 'c1'].day, 11)

    def test_datetime_column_parsed_from_string(self):
        dates = ["2018-01-02", "2018-02-01"]
        df = pd.DataFrame({'c1': dates, 'c2': [3,4]})

        file_name = get_temp_file('.csv')
        df.to_csv(file_name)
        df = pd.read_csv(file_name, parse_dates=['c1'], index_col=0)

        self.assertEqual(df.dtypes[0], np.dtype('datetime64[ns]'))

        pipeline = Pipeline(steps=[Handler(columns={'c2': 'c2'})])
        result = pipeline.fit_transform(df)

        self.assertEqual(result.loc[0, 'c1'].year, 2018)
        self.assertEqual(result.loc[0, 'c1'].month, 1)
        self.assertEqual(result.loc[0, 'c1'].day, 2)
        self.assertEqual(result.loc[0, 'c1'].hour, 0)
        self.assertEqual(result.loc[0, 'c1'].minute, 0)
        self.assertEqual(result.loc[0, 'c1'].second, 0)

        self.assertEqual(result.loc[1, 'c1'].year, 2018)
        self.assertEqual(result.loc[1, 'c1'].month, 2)
        self.assertEqual(result.loc[1, 'c1'].day, 1)
        self.assertEqual(result.loc[1, 'c1'].hour, 0)
        self.assertEqual(result.loc[1, 'c1'].minute, 0)
        self.assertEqual(result.loc[1, 'c1'].second, 0)

        self.assertEqual(len(result), 2)
        self.assertEqual(result.dtypes[0], np.dtype('datetime64[ns]'))

        os.remove(file_name)

    @unittest.skipIf(sys.version_info[:2] == (2, 7), "azureml-dataprep is not installed.")
    def test_dprep_datastream(self):
        import azureml.dataprep as dprep

        dates = ["2018-01-02 00:00:00", "2018-02-01 10:00:00"]
        col2 = ['0', '1']
        label_array = np.repeat([0], 2)
        train_df = pd.DataFrame({'col1': dates, 'col2': col2, 'label': label_array})

        pipeline = Pipeline(steps=[
            Handler(columns={'2': 'col2'}, concat=False, impute_by_slot=True, replace_with='Mean')
        ])

        file_name = get_temp_file('.csv')
        train_df.to_csv(file_name)

        dataflow = dprep.read_csv(file_name, infer_column_types=True)
        dprepDataStream = DprepDataStream(dataflow)

        result = pipeline.fit_transform(dprepDataStream)

        self.assertEqual(result.loc[:, 'col1'].dtype, np.dtype('datetime64[ns]'))

        self.assertEqual(result.loc[0, 'col1'].year, 2018)
        self.assertEqual(result.loc[0, 'col1'].month, 1)
        self.assertEqual(result.loc[0, 'col1'].day, 2)
        self.assertEqual(result.loc[0, 'col1'].hour, 0)
        self.assertEqual(result.loc[0, 'col1'].minute, 0)
        self.assertEqual(result.loc[0, 'col1'].second, 0)

        self.assertEqual(result.loc[1, 'col1'].year, 2018)
        self.assertEqual(result.loc[1, 'col1'].month, 2)
        self.assertEqual(result.loc[1, 'col1'].day, 1)
        self.assertEqual(result.loc[1, 'col1'].hour, 10)
        self.assertEqual(result.loc[1, 'col1'].minute, 0)
        self.assertEqual(result.loc[1, 'col1'].second, 0)

        os.remove(file_name)


if __name__ == '__main__':
    unittest.main()
