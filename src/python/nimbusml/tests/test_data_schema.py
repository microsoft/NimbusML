# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
import sys
import unittest
from collections import OrderedDict

import numpy
import pandas
import six
from nimbusml import DataSchema
from nimbusml import Pipeline, FileDataStream
from nimbusml.datasets import get_dataset
from nimbusml.ensemble import LightGbmRegressor
from nimbusml.feature_extraction.categorical import OneHotVectorizer
from nimbusml.internal.utils.data_roles import DataRoles
from nimbusml.internal.utils.data_schema import DataColumn
from numpy import matrix
from pandas import DataFrame
from scipy.sparse import csr_matrix

if six.PY2:
    from StringIO import StringIO
    from io import open
else:
    from io import StringIO


class TestDataSchema(unittest.TestCase):

    def test_data_roles(self):
        roles = DataRoles(dict(Label='Lab'))
        clone = roles.clone()
        assert roles == clone
        roles._set_role('Feature', 'Feat')
        assert len(roles) == 2
        assert str(list(sorted(roles.items()))
                   ) == "[('Feature', 'Feat'), ('Label', 'Lab')]"
        assert roles._has_role('Feature')
        assert not roles._has_role('Features')
        assert roles._get_role('Feature') == 'Feat'

    def test_data_column(self):

        c1 = DataColumn('col=text:TX:5')
        c2 = DataColumn(name='text', type='TX', pos=5)
        assert str(c1) == 'col=text:TX:5'
        assert str(c2) == 'col=text:TX:5'
        assert c1.__eq__(c2)
        assert c1 == c2
        assert c1 == DataColumn(c2)
        assert not c1.IsVector

        c1 = DataColumn('col=text:TX:2,5-8')
        assert c1.IsVector
        assert str(c1) == 'col=text:TX:2,5-8'

    def test_data_schema_collapse_no(self):

        df = pandas.DataFrame(dict(tt=['a', 'b', 'cc', 'dd', 'ee']))
        df['ff'] = 0.2
        df['ff2'] = 0.1
        df['tt1'] = 'rt'
        df['ii'] = 5
        df['gg'] = 3.4
        st = StringIO()
        df.to_csv(st, index=False)
        st = StringIO(st.getvalue())
        df = pandas.read_csv(st)
        sch = DataSchema.read_schema(df)
        s = str(sch)
        self.assertEqual(
            s,
            'col=tt:TX:0 col=ff:R8:1 col=ff2:R8:2 col=tt1:TX:3 '
            'col=ii:I8:4 col=gg:R8:5 quote+ header=+')

    def test_data_schema_collapse_yes(self):

        df = pandas.DataFrame(dict(tt=['a', 'b', 'cc', 'dd', 'ee']))
        df['ff'] = 0.2
        df['ff2'] = 0.1
        df['tt1'] = 'rt'
        df['ii'] = 5
        df['gg'] = 3.4
        st = StringIO()
        df.to_csv(st, index=False)
        st = StringIO(st.getvalue())
        df = pandas.read_csv(st)
        sch = DataSchema.read_schema(df, collapse=True)
        s = str(sch)
        self.assertEqual(
            s,
            'col=tt:TX:0 col=ff:R8:1-2 col=tt1:TX:3 col=ii:I8:4 '
            'col=gg:R8:5 quote+ header=+')

    def test_data_schema_collapse_no_file(self):

        df = pandas.DataFrame(dict(tt=['a', 'b', 'cc', 'dd', 'ee']))
        df['ff'] = 0.2
        df['ff2'] = 0.1
        df['tt1'] = 'rt'
        df['ii'] = 5
        df['gg'] = 3.4
        st = StringIO()
        df.to_csv(st, index=False)
        st = StringIO(st.getvalue())
        sch = DataSchema.read_schema(st)
        s = str(sch)
        self.assertEqual(
            s,
            'col=tt:TX:0 col=ff:R8:1 col=ff2:R8:2 col=tt1:TX:3 '
            'col=ii:I8:4 col=gg:R8:5 quote+ header=+')

    def test_data_schema_collapse_yes_file(self):

        df = pandas.DataFrame(dict(tt=['a', 'b', 'cc', 'dd', 'ee']))
        df['ff'] = 0.2
        df['ff2'] = 0.1
        df['tt1'] = 'rt'
        df['ii'] = 5
        df['gg'] = 3.4
        st = StringIO()
        df.to_csv(st, index=False)
        st = StringIO(st.getvalue())
        sch = DataSchema.read_schema(st, collapse=True)
        s = str(sch)
        self.assertEqual(
            s,
            'col=tt:TX:0 col=ff:R8:1-2 col=tt1:TX:3 col=ii:I8:4 '
            'col=gg:R8:5 quote+ header=+')

    @unittest.skip(
        reason="needs another entrypoint to guess the schema with nimbusml, "
               "loader does not accept empty schema")
    def test_data_schema_collapse_no_file_loader(self):

        df = pandas.DataFrame(dict(tt=['a', 'b', 'cc', 'dd', 'ee']))
        df['ff'] = 0.2
        df['ff2'] = 0.1
        df['tt1'] = 'rt'
        df['ii'] = 5
        df['gg'] = 3.4
        st = StringIO()
        df.to_csv(st, index=False)
        st = StringIO(st.getvalue())
        sch = DataSchema.read_schema(st, tool='nimbusml')
        s = str(sch)
        self.assertEqual(
            s,
            'col=tt:TX:0 col=ff:R8:1 col=ff2:R8:2 col=tt1:TX:3 '
            'col=ii:I8:4 col=gg:R8:5 header=+')

    @unittest.skip(
        reason="needs another entrypoint to guess the schema with nimbusml, "
               "loader does not accept empty schema")
    def test_data_schema_collapse_yes_file_loader(self):

        df = pandas.DataFrame(dict(tt=['a', 'b', 'cc', 'dd', 'ee']))
        df['ff'] = 0.2
        df['ff2'] = 0.1
        df['tt1'] = 'rt'
        df['ii'] = 5
        df['gg'] = 3.4
        st = StringIO()
        df.to_csv(st, index=False)
        st = StringIO(st.getvalue())
        sch = DataSchema.read_schema(st, collapse=True, tool='nimbusml')
        s = str(sch)
        self.assertEqual(
            s,
            'col=ff:R8:1-2 col=gg:R8:5 col=ii:I8:4 col=tt:TX:0 '
            'col=tt1:TX:3 header=+')

    def test_data_schema(self):
        s0 = DataSchema('col=text:TX:5')
        s1 = DataSchema([DataColumn('col=text:TX:5')])
        s2 = DataSchema([DataColumn(name='text', type='TX', pos=5)])
        assert list(s0.columns.keys()) == ['text']
        assert list(s1.columns.keys()) == ['text']
        assert str(s1) == 'col=text:TX:5 quote+'
        assert str(s2) == 'col=text:TX:5 quote+'
        assert str(s0) == 'col=text:TX:5 quote+'
        assert s1 == s2
        assert s1 == s0
        assert s1 == DataSchema(s0)
        assert s1[0] == s1['text']
        assert s1 == s1[['text']]
        assert len(s1) == 1
        assert len(list(s1)) == 1
        assert 'text' in s1
        assert 'tex' not in s1

    def test_data_schema_read_schema(self):
        df = pandas.DataFrame(dict(a=[0, 1], b=[0.1, 1.1], c=['r', 'd'],
                                   d=[False, True]))
        sch = DataSchema.read_schema(df)
        assert str(
            sch) == 'col=a:I8:0 col=b:R8:1 col=c:TX:2 col=d:BL:3 quote+ header=+'
        sch = DataSchema.read_schema(df, sep=',')
        assert str(
            sch) == 'col=a:I8:0 col=b:R8:1 col=c:TX:2 col=d:BL:3 ' \
                    'quote+ header=+ sep=,'
        csr = csr_matrix([[0, 1], [1, 0]], dtype='int32')
        sch = DataSchema.read_schema(csr, sep=',')
        assert str(sch) == 'col=Data:I4:0-1 quote+ header=+ sep=,'
        csr = matrix([[0, 1], [1, 0]], dtype='int32')
        sch = DataSchema.read_schema(csr, sep=',')
        assert str(sch) == 'col=Data:I4:0-1 quote+ header=+ sep=,'
        csr = matrix([[0, 1], [1.5, 0.5]])
        sch = DataSchema.read_schema(csr, sep=',')
        assert str(sch) == 'col=Data:R8:0-1 quote+ header=+ sep=,'

    def test_data_schema_read_schema_tab(self):
        df = pandas.DataFrame(dict(a=[0, 1], b=[0.1, 1.1], c=['r', 'd'],
                                   d=[False, True]))
        sch = DataSchema.read_schema(df)
        assert str(
            sch) == 'col=a:I8:0 col=b:R8:1 col=c:TX:2 col=d:BL:3 quote+ header=+'
        sch = DataSchema.read_schema(df, sep='\t')
        assert str(
            sch) == 'col=a:I8:0 col=b:R8:1 col=c:TX:2 col=d:BL:3 ' \
                    'quote+ header=+ sep=tab'

    def test_schema_infert(self):
        train_file = get_dataset("infert").as_filepath()
        found = DataSchema.read_schema(train_file)
        schema = "col=row_num:I8:0 col=education:TX:1 col=age:I8:2 " \
                 "col=parity:I8:3 col=induced:I8:4 " + \
                 "col=case:I8:5 col=spontaneous:I8:6 col=stratum:I8:7 " \
                 "col=pooled.stratum:I8:8 quote+ header=+"
        assert str(found) == schema
        fds = FileDataStream(train_file, schema)
        assert str(fds.schema) == schema
        fds = FileDataStream.read_csv(train_file)
        assert str(fds.schema) == schema

    def test_schema_infert_R4(self):
        train_file = get_dataset("infert").as_filepath()
        found = DataSchema.read_schema(train_file,
                                       numeric_dtype=numpy.float32)
        schema = "col=row_num:R4:0 col=education:TX:1 col=age:R4:2 " \
                 "col=parity:R4:3 col=induced:R4:4 " + \
                 "col=case:R4:5 col=spontaneous:R4:6 col=stratum:R4:7 " \
                 "col=pooled.stratum:R4:8 quote+ header=+"
        assert str(found) == schema
        fds = FileDataStream(train_file, schema)
        assert str(fds.schema) == schema
        fds = FileDataStream.read_csv(train_file,
                                      numeric_dtype=numpy.float32)
        assert str(fds.schema) == schema

    def test_schema_infert_R4one(self):
        train_file = get_dataset("infert").as_filepath()
        found = DataSchema.read_schema(
            train_file, dtype={'age': numpy.float32})
        schema = "col=row_num:I8:0 col=education:TX:1 col=age:R4:2 " \
                 "col=parity:I8:3 col=induced:I8:4 " + \
                 "col=case:I8:5 col=spontaneous:I8:6 col=stratum:I8:7 " \
                 "col=pooled.stratum:I8:8 quote+ header=+"
        assert str(found) == schema
        fds = FileDataStream(train_file, schema)
        assert str(fds.schema) == schema
        fds = FileDataStream.read_csv(train_file,
                                      dtype={'age': numpy.float32})
        assert str(fds.schema) == schema

    def test_schema_airquality(self):
        train_file = get_dataset("airquality").as_filepath()
        found = DataSchema.read_schema(train_file)
        schema = "col=Unnamed0:I8:0 col=Ozone:R8:1 col=Solar_R:R8:2 " \
                 "col=Wind:R8:3 col=Temp:I8:4 col=Month:I8:5 " \
                 "col=Day:I8:6 quote+ header=+"
        assert str(found) == schema
        fds = FileDataStream(train_file, schema)
        assert str(fds.schema) == schema
        fds = FileDataStream.read_csv(train_file)
        assert str(fds.schema) == schema

    def test_schema_collapse_all(self):
        path = get_dataset('infert').as_filepath()

        file_schema = DataSchema.read_schema(path, collapse='all', sep=',',
                                             numeric_dtype=numpy.float32,
                                             names={0: 'row_num',
                                                    5: 'case'})
        file_schema.rename('age', 'Features')
        assert str(
            file_schema) == "col=row_num:R4:0 col=education:TX:1 " \
                            "col=Features:R4:2-4,6-8 col=case:R4:5 " \
                            "quote+ header=+ sep=,"

    def test_schema_documentation(self):

        data = DataFrame(
            OrderedDict(
                real=[
                    0.1, 0.2], integer=[
                    1, 2], text=[
                    "a", "b"]))
        data['real32'] = data['real'].astype(numpy.float32)
        schema = DataSchema.read_schema(data)
        if sys.version_info[:2] >= (3, 6):
            assert str(
                schema) == 'col=real:R8:0 col=integer:I8:1 col=text:TX:2 ' \
                           'col=real32:R4:3 quote+ header=+'

        data = DataFrame(
            OrderedDict(
                real=[
                    0.1, 0.2], integer=[
                    1, 2], text=[
                    "a", "b"]))
        data.to_csv('data.txt', index=False)
        schema = DataSchema.read_schema('data.txt')
        if sys.version_info[:2] >= (3, 6):
            assert str(
                schema) == 'col=real:R8:0 col=integer:I8:1 col=text:TX:2' \
                           ' quote+ header=+'

        data = DataFrame(
            OrderedDict(
                real=[
                    0.1, 0.2], integer=[
                    1, 2], text=[
                    "a", "b"]))
        data.to_csv('data.txt', index=False)
        schema = DataSchema.read_schema('data.txt')
        if sys.version_info[:2] >= (3, 6):
            assert str(
                schema) == 'col=real:R8:0 col=integer:I8:1 col=text:TX:2' \
                           ' quote+ header=+'

        data = DataFrame(
            OrderedDict(
                real=[
                    0.1, 0.2], real2=[
                    0.1, 0.2], integer=[
                    1, 2], text=[
                    "a", "b"]))
        data.to_csv('data.txt', index=False)
        schema = DataSchema.read_schema('data.txt', collapse=True)
        if sys.version_info[:2] >= (3, 6):
            assert str(
                schema) == 'col=real:R8:0-1 col=integer:I8:2 ' \
                           'col=text:TX:3 quote+ header=+'

        data = DataFrame(
            OrderedDict(
                real=[
                    0.1, 0.2], text1=[
                    "a", "b"], text2=[
                    "a", "b"]))
        data.to_csv('data.txt', index=False)
        schema = DataSchema.read_schema('data.txt', collapse=True,
                                        names={0: 'newname',
                                               1: 'newname2'})
        if sys.version_info[:2] >= (3, 6):
            assert str(
                schema) == 'col=newname:R8:0 col=newname2:TX:1-2 quote+ header=+'

        data = DataFrame(
            OrderedDict(
                real=[
                    0.1, 0.2], text1=[
                    "a", "b"], text2=[
                    "a", "b"]))
        data.to_csv('data.txt', index=False)
        schema = DataSchema.read_schema(
            'data.txt', collapse=False, names={(1, None): 'text'})
        if sys.version_info[:2] >= (3, 6):
            assert str(
                schema) == 'col=real:R8:0 col=text_0:TX:1 ' \
                           'col=text_1:TX:2 quote+ header=+'

        data = DataFrame(OrderedDict(real=[0.1, 0.2], text1=["a", "b"]))
        data.to_csv('data.txt', index=False)
        schema = DataSchema.read_schema(
            'data.txt', collapse=True, dtype={
                'real': numpy.float32})
        if sys.version_info[:2] >= (3, 6):
            assert str(schema) == 'col=real:R4:0 col=text1:TX:1 quote+ header=+'
        for c in schema:
            assert repr(c).startswith("DataColumn(name='")
        assert repr(schema).startswith("DataSchema([DataColumn(name='")

    def test_schema_tab(self):
        train_file = get_dataset('topics').as_filepath()

        train_file_stream = FileDataStream.read_csv(
            train_file, sep=',', names={
                0: 'review', 1: 'review_reverse', 2: 'label'})
        with open(train_file, 'r', encoding='utf-8') as f:
            first_line = f.readline()
        header = first_line.strip(' \n\r').split(',')

        assert header == ['review', 'review_reverse', 'label']
        print(str(train_file_stream.schema))
        assert str(
            train_file_stream.schema) == 'col=review:TX:0 ' \
                                         'col=review_reverse:TX:1 ' \
                                         'col=label:I8:2 quote+ header=+ sep=,'

        train_file_stream = FileDataStream.read_csv(
            train_file, sep=',', names={
                0: 'review', 1: 'review_reverse', 2: 'label'}, dtype={
                'label': numpy.uint32})
        assert str(
            train_file_stream.schema) == 'col=review:TX:0 ' \
                                         'col=review_reverse:TX:1 ' \
                                         'col=label:U4:2 quote+ header=+ sep=,'

    def test_schema_dtype_regex(self):
        path = get_dataset('gen_tickettrain').as_filepath()
        file_schema = DataSchema.read_schema(
            path,
            collapse='all',
            sep=',',
            names={
                0: 'Label',
                1: 'GroupId',
                2: 'carrier',
                (3,
                 None): 'Features'},
            dtype={
                'GroupId': str,
                'Label': numpy.float32,
                'carrier': str,
                'Features_[0-9]{1,2}': numpy.float32})
        file_schema.rename('Features_0', 'Features')
        assert str(
            file_schema) == 'col=Label:R4:0 col=GroupId:TX:1 ' \
                            'col=carrier:TX:2 col=Features:R4:3-7 ' \
                            'quote+ header=+ sep=,'

    def test_schema_dtype_slice(self):
        path = get_dataset('gen_tickettrain').as_filepath()
        file_schema = DataSchema.read_schema(
            path, sep=',', collapse='all', names={
                0: 'Label', 1: 'GroupId'}, dtype={
                'GroupId': str, 'Label': numpy.float32, 'carrier': str,
                'price': numpy.float32})
        assert str(
            file_schema) == 'col=Label:R4:0 col=GroupId:TX:1 ' \
                            'col=carrier:TX:2 col=price:R4:3 ' \
                            'col=Class:I8:4-6 col=duration:R8:7 quote+ header=+ ' \
                            'sep=,'

    def test_schema_dtype_list_int(self):
        li = [[1.0, 1.0, 2.0], [3.0, 5.0, 6.0]]
        schema = DataSchema.read_schema(li)
        assert str(
            schema) == 'col=c0:R8:0 col=c1:R8:1 col=c2:R8:2 quote+ header=+'

    def test_schema_dtype_list_trueint(self):
        li = [[1, 1, 2], [3, 5, 6]]
        schema = DataSchema.read_schema(li)
        assert str(
            schema) == 'col=c0:I8:0 col=c1:I8:1 col=c2:I8:2 quote+ header=+'

    def test_schema_dtype_numpy_trueint(self):
        li = [[1, 1, 2], [3, 5, 6]]
        mat = numpy.array(li)
        dt = mat.dtype
        schema = DataSchema.read_schema(mat)
        # The behavior is not the same on every OS.
        if dt == numpy.int64:
            assert str(schema) == 'col=Data:I8:0-2 quote+ header=+'
        elif dt == numpy.int32:
            assert str(schema) == 'col=Data:I4:0-2 quote+ header=+'
        else:
            raise TypeError("unexpected type {0}".format(dt))

    def test_schema_dtype_numpy_float(self):
        li = [[1.0, 1.0, 2.0], [3.0, 5.0, 6.0]]
        mat = numpy.array(li)
        schema = DataSchema.read_schema(mat)
        assert str(schema) == 'col=Data:R8:0-2 quote+ header=+'

    def test_schema_sep_default(self):
        data = pandas.DataFrame(
            dict(
                real=[
                    0.1, 2.2], text=[
                    'word', 'class'], y=[
                    1, 3]))
        data.to_csv('data.csv', index=False, header=True)
        ds = FileDataStream.read_csv(
            'data.csv',
            collapse=False,
            numeric_dtype=numpy.float32)
        assert str(
            ds.schema) == "col=real:R4:0 col=text:TX:1 col=y:R4:2 quote+ header=+"
        assert ds.schema.to_string() == "col=real:R4:0 col=text:TX:1 " \
                                        "col=y:R4:2 quote+ header=+"
        assert ds.schema.to_string(
            add_sep=True) == "col=real:R4:0 col=text:TX:1 col=y:R4:2 " \
                             "quote+ header=+ sep=,"
        exp = Pipeline([OneHotVectorizer(columns=['text']),
                        LightGbmRegressor(minimum_example_count_per_leaf=1)])
        exp.fit(ds, 'y')
        pred = exp.predict(ds)
        assert pred is not None
        assert len(pred) > 0

    def test_schema__repr(self):
        path = get_dataset('infert').as_filepath()
        data = FileDataStream.read_csv(
            path, sep=',', numeric_dtype=numpy.float32)
        assert str(
            data.schema) == "col=row_num:R4:0 col=education:TX:1 " \
                            "col=age:R4:2 col=parity:R4:3 " \
                            "col=induced:R4:4 col=case:R4:5 " \
                            "col=spontaneous:R4:6 col=stratum:R4:7 " \
                            "col=pooled.stratum:R4:8 quote+ header=+ sep=,"
        assert "DataSchema([DataColumn(name='row_num', type='R4', " \
               "pos=0)" in str(repr(data.schema))

        path = get_dataset('topics').as_filepath()
        data = FileDataStream.read_csv(
            path, sep=',', numeric_dtype=numpy.float32, collapse=True)
        assert str(
            data.schema) == "col=review:TX:0-1 col=label:R4:2 quote+ header=+ " \
                            "sep=,"
        assert "DataSchema([DataColumn(name='review', type='TX', pos=(0," \
               " 1))" in str(repr(data.schema))

        path = get_dataset('topics').as_filepath()
        data = FileDataStream.read_csv(
            path, sep=',', numeric_dtype=numpy.float32, collapse=False)
        assert str(
            data.schema) == "col=review:TX:0 col=review_reverse:TX:1 " \
                            "col=label:R4:2 quote+ header=+ sep=,"
        assert "DataSchema([DataColumn(name='review', type='TX', pos=0)," \
               in str(repr(data.schema))


if __name__ == "__main__":
    unittest.main()
