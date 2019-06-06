# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
import os
import tempfile
import unittest

import numpy
import pandas
from nimbusml import DataSchema
from nimbusml import FileDataStream

try:
    from pandas.testing import assert_frame_equal
except ImportError:
    # earlier versions
    from pandas.util.testing import assert_frame_equal


class TestDataStream(unittest.TestCase):

    def test_data_stream(self):
        df = pandas.DataFrame(dict(a=[0, 1], b=[0.1, 0.2]))
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            df.to_csv(f, sep=',', index=False)

        fi = FileDataStream.read_csv(f.name, sep=',')
        fi2 = fi.clone()
        assert repr(fi) == repr(fi2)
        os.remove(f.name)

    def test_data_header_no_dataframe(self):
        li = [1.0, 1.0, 2.0]
        df = pandas.DataFrame(li)
        schema0 = DataSchema.read_schema(df)
        assert str(schema0) == 'col=c0:R8:0 quote+ header=+'

        li = [[1.0, 1.0, 2.0], [3.0, 5.0, 6.0]]
        schema1 = DataSchema.read_schema(li)
        assert str(schema1) == 'col=c0:R8:0 col=c1:R8:1 col=c2:R8:2 quote+ header=+'

        df = pandas.DataFrame([[1.0, 1.0, 2.0], [3.0, 5.0, 6.0]])
        schema2 = DataSchema.read_schema(df)
        assert str(schema2) == 'col=c0:R8:0 col=c1:R8:1 col=c2:R8:2 quote+ header=+'

        mat = numpy.array([[1.0, 1.0, 2.0], [3.0, 5.0, 6.0]])
        schema3 = DataSchema.read_schema(mat)
        assert str(schema3) == 'col=Data:R8:0-2 quote+ header=+'

        li = [1.0, 1.0, 2.0]
        df = pandas.DataFrame(li)
        schema0 = DataSchema.read_schema(df, header=False)
        assert str(schema0) == 'col=c0:R8:0 quote+ header=-'

    def test_data_stream_head_file(self):
        df = pandas.DataFrame(dict(a=[0, 1], b=[0.1, 0.2]))
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            df.to_csv(f, sep=',', index=False)

        df1 = df.head(1)
        df2 = df[1:].reset_index(drop=True)

        fi = FileDataStream.read_csv(f.name, sep=',')
        head = fi.head(1)
        head2 = fi.head(1, 1)
        assert_frame_equal(head, df1)
        assert_frame_equal(head2, df2)
        head3 = fi.head(1, 1, collect=False).transform(fi, verbose=0)
        assert_frame_equal(head3, df2)

        dff = fi.to_df()
        assert_frame_equal(df, dff)

        os.remove(f.name)


if __name__ == "__main__":
    unittest.main()
