# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
"""
Schema definition.
"""
import copy
import os
import re
import sys
import tempfile
from collections import OrderedDict
from textwrap import wrap

import six
from numpy import ndarray
from pandas import DataFrame, Series, read_csv
from scipy.sparse import csr_matrix
from six import text_type

if six.PY2:
    from StringIO import StringIO
else:
    from io import StringIO


def update_dict(a, b):
    c = a.copy()
    c.update(b)
    return c


class DataColumn:
    """
    Defines a column for a datasets, it can be a single
    value or a vector of values.
    """
    _type_mapping_exact = {
        'bool': 'BL',
        'int8': 'I1',
        'int16': 'I2',
        'int32': 'I4',
        'int64': 'I8',
        'uint8': 'U1',
        'uint16': 'U2',
        'uint32': 'U4',
        'uint64': 'U8',
        'float32': 'R4',
        'float64': 'R8',
        'object': 'TX',
        'str': 'TX',
        '<type \'str\'>': 'TX'}

    @staticmethod
    def get_type_mapping():
        """
        Returns a dictionary which contains maps numpy.dtype to nimbusml types.
        """
        return DataColumn._type_mapping_exact

    def __init__(self, scol=None, **kwargs):
        """
        REVIEW: remove features.
        :param scol: column defined as a string
        :param kwargs: additional fields such as name, type, pos, length

        Both parameters are exclusive. A column cannot be defined
        with both at the same time.
        """
        if scol is not None:
            if isinstance(scol, str):
                self.parse(scol)
            elif isinstance(scol, DataColumn):
                self.name = scol.name
                self.type = scol.type
                self.pos = scol.pos
            else:
                raise TypeError('Unable to handle {0}'.format(scol))
        elif isinstance(scol, tuple):
            raise TypeError("scol cannot be a tuple, only y a string.")
        else:
            self.name = kwargs.get('name', 'Data')
            dtype = kwargs['type']
            if not isinstance(dtype, str):
                dtype = DataColumn.type2str(dtype)
            self.type = DataColumn.get_type_mapping().get(dtype, dtype)
            pos = kwargs.get('pos', 0)
            length = kwargs.get('length', None)
            if isinstance(pos, six.integer_types):
                if length is None:
                    self.pos = pos
                else:
                    self.pos = tuple(list(range(pos, pos + length)))
            elif isinstance(pos, (tuple, list)):
                self.pos = tuple(pos)
            else:
                raise TypeError('Unable to handle {0}'.format(kwargs))
        self.check_name()

    def rename(self, new_name):
        """
        Renames the column.

        :param new_name: new name
        :return: self
        """
        self.name = new_name
        self.check_name()
        return self

    @staticmethod
    def type2str(t):
        """
        Converts a type into a string.
        """
        s = str(t)
        if 'numpy' in s:
            s = t.__name__
        elif '<class' in s:
            if t == int:
                s = 'int64'
            elif t == str:
                s = 'str'
            elif s == 'float':
                s = 'R8'
            else:
                raise ValueError(
                    "Unable to convery type {0} into a string".format(t))
        return s

    def check_name(self):
        """
        Check the name is consistent with *nimbusml*.
        """
        if not isinstance(self.name, (str, text_type)):
            raise TypeError(
                "name must be a string not {0}".format(
                    type(
                        self.name)))
        if ' ' in self.name:
            raise ValueError(
                "Spaces are not allowed in '{0}'".format(
                    self.name))
        if '\n' in self.name:
            raise ValueError("EOL are not allowed in '{0}'".format(self.name))
        if '\r' in self.name:
            raise ValueError("\\r are not allowed in '{0}'".format(self.name))
        if '\t' in self.name:
            raise ValueError("tabs are not allowed in '{0}'".format(self.name))
        if ':' in self.name:
            raise ValueError(": are not allowed in '{0}'".format(self.name))

    def clone(self):
        return copy.copy(self)

    def is_numeric(self):
        """
        Tells if a column is numeric.
        """
        return self.type != 'TX'

    def change_type(self, new_type):
        """
        Changes the type, keep every else unchanged.
        """
        if new_type is None:
            raise ValueError("new_type cannot be None")
        c = self.clone()
        if not isinstance(new_type, str):
            new_type = DataColumn.type2str(new_type)
        c.type = DataColumn.get_type_mapping().get(new_type, new_type)
        return c

    @property
    def Name(self):
        return self.name

    @property
    def Type(self):
        return self.type

    @property
    def Pos(self):
        return self.pos

    @property
    def IsVector(self):
        return not isinstance(self.pos, six.integer_types)

    def __eq__(self, other):
        return self.name == other.name and \
               self.pos == other.pos and \
               self.type == other.type

    def format_pos(self):
        if isinstance(self.pos, six.integer_types):
            return str(self.pos)
        else:
            begin = self.pos[0]
            last = self.pos[0]
            res = []
            for i in range(1, len(self.pos)):
                if self.pos[i] == last + 1:
                    last += 1
                elif begin == last:
                    res.append(str(begin))
                    begin = last = self.pos[i]
                else:
                    res.append("%d-%d" % (begin, last))
                    begin = last = self.pos[i]

            if begin == last:
                res.append(str(begin))
            else:
                res.append("%d-%d" % (begin, last))
            return ",".join(res)

    @property
    def name_as_string(self):
        """
        A tuple is used to specify a column in a multilevel index
        in a pandas dataframe. This function replaces it by a
        unique string where ``'.'`` separates the pieces of the tuple.
        """
        if isinstance(self.name, tuple):
            # tuple for multilvel index from pandas dataframe
            not_none = [str(_) for _ in self.name if _ is not None]
            return '.'.join(not_none)
        else:
            return self.name

    def __str__(self):
        return "col=%s:%s:%s" % (
            self.name_as_string, self.type, self.format_pos())

    def __repr__(self):
        if isinstance(self.pos, six.integer_types):
            rpos = self.pos
        elif len(self.pos) == 1:
            rpos = self.pos[0]
        else:
            rpos = self.pos
        return "DataColumn(name='{}', type='{}', pos={})".format(
            self.name, self.type, rpos)

    def parse(self, column):
        """
        Parses a column definition as text.
        """
        if column.startswith('col='):
            column = column[4:]
        if '=' in column:
            raise ValueError(
                "Wrong format for a column definition '{0}'".format(column))
        spl = column.split(':')
        if len(spl) != 3:
            raise ValueError(
                "Wrong format for a column definition '{0}'".format(column))
        self.name = spl[0]
        self.type = spl[1]
        pos = spl[2]
        cols = []
        for p in pos.split(','):
            if '-' in p:
                spl = p.split('-')
                if len(spl) != 2:
                    raise NotImplementedError(
                        "Unable to parse '{0}' yet".format(column))
                cols.extend(range(int(spl[0]), int(spl[1]) + 1))
            else:
                cols.append(int(p))
        if len(cols) == 1:
            self.pos = cols[0]
        else:
            self.pos = tuple(cols)

    def __lt__(self, o):
        """
        So that lists of DataColumn can be sorted.
        """
        o1 = self.pos if isinstance(self.pos, six.integer_types) else self.pos[0]  # tuple
        o2 = o.pos if isinstance(o.pos, six.integer_types) else o.pos[0]  # tuple
        return o1 < o2


class DataSchema:
    """

    Defines a schema for a datasets.

    .. remarks::
        The DataSchema class automatically generates a description of
        the data schema from various data sources. The
        data source may be a list, array, dataframe or a file. A schema
        is required for all ``nimbusml`` trainers and
        transforms, and when not provided explicitly, it needs to be
        inferred automatically before any data processing
        can occur. In the case of list, array or dataframes, the schema
        inference is usually straightforward, but when
        the data source is a file, it may require further inspection to
        ensure it matches the data, and that the types
        are aligned as needed (e.g. R4 vs I4).

        For more details on the schema format, refer to
        `Schema </nimbusml/concepts/schema#dataschema-class>`_,
        `Types </nimbusml/concepts/types#column-types>`_
        and
        `Vector Type </nimbusml/concepts/types#vectortype-column>`_.

    .. seealso::
        :py:func:`FileDataStream <nimbusml.FileDataStream>`.

    Example:
        .. code-block:: python

            from nimbusml import DataSchema, FileDataStream
            from nimbusml import Pipeline
            from nimbusml.ensemble import LightGbmRegressor
            from nimbusml.feature_extraction.categorical import OneHotVectorizer
            import numpy as np
            import pandas as pd

            data = pd.DataFrame(dict(real = [0.1, 2.2],
                                    text = ['word','class'],
                                    y = [1,3]))
            data.to_csv('data.csv', index = False, header = True)

            schema = DataSchema.read_schema('data.csv', collapse = False,
                                            numeric_dtype = np.float32,
                                            sep = ',')
            print(schema)
            #col=real:R4:0 col=text:TX:1 col=y:R4:2 header=+ sep=,

            exp = Pipeline([
                         OneHotVectorizer(columns = ['text']),
                         LightGbmRegressor(minimum_example_count_per_leaf = 1)
                        ])

            exp.fit(FileDataStream('data.csv', schema = schema), 'y')

    """
    _default_options = dict(sep=',')

    def __init__(self, schema, **options):
        """
        :param schema: data schema (string)
        :param options: additional options

        Roles defines how columns are used if not overwritten
        by a trainer or a transform. Roles can be
        label, feature, group, weight.
        """
        if schema is None:
            raise ValueError("The schema must be specified.")
        if isinstance(schema, str):
            self.parse(schema)
        elif isinstance(schema, list):
            self.columns = OrderedDict()
            for c in schema:
                self.columns[c.Name] = c
            self.options = {}
        elif isinstance(schema, DataSchema):
            self.columns = schema.columns
            self.options = schema.options
        else:
            raise TypeError("Unable to handle type {0}".format(type(schema)))
        self.options.update(options)
        if 'sep' in self.options and self.options['sep'] == 'tab':
            self.options['sep'] = "\t"
        self._check_options()

    def _check_options(self):
        if 'header' in self.options:
            if not isinstance(self.options['header'], bool):
                raise TypeError(
                    'header must be a boolean not {0}'.format(
                        type(
                            self.options['header'])))
        if 'collapse' in self.options:
            raise ValueError("collapse is not allowed here")
        if 'numeric_dtype' in self.options:
            raise ValueError("numeric_dtype is not allowed here")

    def clone(self):
        return DataSchema(list(c.clone() for c in self.columns.values()),
                          **self.options)

    def __len__(self):
        return len(self.columns)

    def __iter__(self):
        """
        Enumerates columns (as DataColumn).
        """
        for k, v in self.columns.items():
            yield v

    def __str__(self):
        return self.to_string()

    def to_string(self, add_sep=False):
        """
        Converts the schema into a string.

        :param add_sep: sep is not added if the user does not specify it,
            but it is required by the core library, the method
            adds the default value if not specified.
        :return: formatted schema as a string
        """
        sch = " ".join(str(c) for n, c in self.columns.items())
        opt = self.format_options(add_sep=add_sep)
        if opt:
            sch += " " + opt
        return sch

    def __repr__(self):
        def display_repr(v):
            if isinstance(v, str):
                return repr([v]).strip('[]')
            elif isinstance(v, tuple) and len(v) == 1:
                return display_repr(v[0])
            else:
                return repr(v).strip("'")

        sch = ", ".join(repr(c) for n, c in self.columns.items())
        opt = ", ".join('{}={}'.format(k, display_repr(v))
                        for k, v in self.options.items())
        return "\n".join(
            wrap(
                "DataSchema([{}], {})".format(
                    sch,
                    opt),
                subsequent_indent='    '))

    def __getitem__(self, i):
        if isinstance(i, six.integer_types):
            # not efficient
            keys = list(self.columns.keys())
            return self.columns[keys[i]]
        elif isinstance(i, str):
            return self.columns[i]
        elif isinstance(i, list):
            cols = [self[ii] for ii in i]
            return DataSchema(cols)
        else:
            raise TypeError(
                "Index can be int, str, list not {0}".format(
                    type(i)))

    def __eq__(self, other):
        if len(self) != len(other):
            return False
        for a, b in zip(self.columns, other.columns):
            if a != b:
                return False
        return self.options == other.options

    def format_options(self, add_sep=False):
        """
        Formats the options for the parser from the core library.

        :param add_sep: the code library usually requires the separator, it
            is not added if the user does not explicitely specify it unless
            *add_sep* is True, in that case, the default value is added.
        :return: formatted options as a string
        """
        opts = self.options
        if add_sep and 'sep' not in self.options:
            opts = opts.copy()
            opts['sep'] = DataSchema._default_options['sep']

        val = ['quote+']
        for k, v in sorted(opts.items()):
            if k == 'quote':
                continue
            if isinstance(v, bool):
                v = "+" if v else '-'
            elif k == 'sep' and v == '\t':
                v = 'tab'
            else:
                v = str(v) if v != '\t' else '\\t'
            val.append('{0}={1}'.format(k, v))
        return " ".join(val)

    def __contains__(self, name):
        """
        Checks that column name belongs to the schema.
        """
        return name in self.columns

    def rename(self, old_name, new_name):
        """
        Renames a column.

        :param old_name: old name
        :param new_name: new_name
        :return: self
        """
        if old_name not in self:
            raise KeyError("Column '{0}' does not exist.".format(old_name))
        self[old_name].rename(new_name)
        return self

    def parse(self, schema):
        """
        Parses a schema defined as a string.
        """
        bool_ = {'+': True, '-': False}
        spl = schema.split(" ")
        self.columns = OrderedDict()
        self.options = {}
        for s in spl:
            if not s:
                continue
            if s.startswith('col='):
                c = DataColumn(s)
                self.columns[c.Name] = c
            else:
                eq = s.split('=')
                if len(eq) > 2:
                    raise RuntimeError("Unable to parse '{0}'".format(schema))
                elif len(eq) == 2:
                    self.options[eq[0]] = bool_.get(eq[1], eq[1])
                else:
                    if s[-1] not in ('-', '+'):
                        raise RuntimeError(
                            "Unable to parse '{0}'".format(schema))
                    self.options[s[:-1]] = bool_.get(s[-1])

    @staticmethod
    def _rename_columns(df, names):
        """
        Renames columns names based on names defined as a dictionary.

        ::

            found = DataSchema.read_schema(train_file, header=False,
                                           names={0:'Label', 1:'GroupId',
                                                 (2,None):'Features'})

        It renames columns 0, 1, as *Label* and *GroupId*.
        It renames columns 2:end with Features_0, ..., Features_2040.
        """
        if not isinstance(names, dict):
            raise TypeError(
                "names must be a dictionary not {0}".format(
                    type(names)))
        columns = list(df.columns)
        for k, v in names.items():
            if isinstance(k, six.integer_types):
                columns[k] = v
            elif isinstance(k, tuple):
                if len(k) != 2:
                    raise ValueError(
                        "k must be (i1, i2), (None, i), (i, None)")
                if k[0] is None:
                    k = (0, k[1])
                if k[1] is None:
                    k = (k[0], len(columns) - 1)
                for i in range(k[0], k[1] + 1):
                    columns[i] = '{0}_{1}'.format(v, i - k[0])
            else:
                raise ValueError("k must be an int or a tuple")
        df.columns = columns

    @staticmethod
    def read_schema_file(
            filepath_or_buffer,
            tool='pandas',
            nrows=100,
            **options):
        """
        Infers the schema of a file.

        :param filepath_or_buffer: stream or filename
        :param tool: `'pandas'` or `'nimbusml'`
        :param nrows: use the first top rows only
        :param options: additional options for *read_csv* from *pandas* or
            internal reader
        :return: schema

        Additional options:

        * collapse: (False by default), collapse columns for of the same type
          if it follows *read_csv* function. Use internal structure of a
          dataframe. If ``collapse* == 'all'``,
          the method collapses all columns not specified in parameter *names*.
        * numeric_dtype: if not None, changes all numeric types into this type
        """
        if tool == 'pandas':
            pd_options = {
                k: v for k, v in options.items() if k not in {
                    'collapse',
                    'numeric_dtype',
                    'features',
                    'drop'}}
            names = pd_options.get('names', None)
            if 'header' in pd_options:
                if isinstance(pd_options['header'], bool):
                    if pd_options['header']:
                        pd_options['header'] = 'infer'
                    else:
                        pd_options['header'] = None
            if isinstance(names, dict):
                # Addition to pandas.
                names = pd_options['names']
                del pd_options['names']
                df = read_csv(filepath_or_buffer, nrows=nrows, **pd_options)
                DataSchema._rename_columns(df, names)
            else:
                df = read_csv(filepath_or_buffer, nrows=nrows, **pd_options)

            # We remove integers as column names.
            df.columns = [_ if isinstance(_, six.string_types)
                          else 'c' + str(_) for _ in df.columns]

            if isinstance(pd_options.get('dtype', None), dict):
                # We overwrite types if specified.
                # It does not seem to be taken into account all the time. It is
                # enforced.
                dtype = pd_options['dtype']
                set_cols = set(df.columns)
                for k, v in dtype.items():
                    if k in set_cols:
                        cols = [k]
                    else:
                        reg = re.compile(k)
                        cols = [
                            _ for _ in df.columns if isinstance(
                                _, str) and reg.search(_)]

                    if len(cols) == 0:
                        first_ten = list(map(str, df.columns))
                        if len(first_ten) > 10:
                            first_ten = first_ten[:10] + ['...']
                        raise ValueError(
                            "Column '{0}' was not found in {1}".format(
                                k, ', '.join(first_ten)))
                    for c in cols:
                        if df[c].dtype != v:
                            df[c] = df[c].astype(v)

            return DataSchema.read_schema(df, **options)

        elif tool == 'nimbusml':
            from ..entrypoints.importtextdata_importtext import data_textloader
            from .entrypoints import Graph
            from .data_stream import FileDataStream

            def handle_file(filename):
                node = data_textloader(
                    input_file="$file", data="$data", **options)
                graph_nodes = [node]
                graph = Graph(*(graph_nodes), inputs=dict(file=filename),
                              outputs=dict(data=''))
                st = FileDataStream(filename, schema=None)
                (out_model, out_data, out_metrics) = graph.run(verbose=True,
                                                               X=st)

            if isinstance(filepath_or_buffer, StringIO):
                with tempfile.NamedTemporaryFile('w', encoding=options.get(
                        'encoding', None), delete=False) as fp:
                    fp.write(filepath_or_buffer.getvalue())
                    filepath_or_buffer = fp.name
                os.remove(filepath_or_buffer)
            else:
                return handle_file(filepath_or_buffer)
        else:
            raise ValueError(
                "Unknown tool '{0}', choose 'pandas' or 'nimbusml'".format(tool))

        # self._check_options() not sure what this is for

    @staticmethod
    def read_schema(*data, **options):
        """
        Infers the schema of a data view.

        :param data: features, labels, weights, groups 
        :param collapse: (False by default), collapse columns for of the same type
          if it follows *read_csv* function. Use internal structure of a
          dataframe. If ``collapse* == 'all'``,
          the method collapses all columns not specified in parameter *names*.
        :param sep: string value of file seperation character (for example: ',')
        :param header: whether the data has a header row; defaults to True 
        :param dtype: change dtype of specific columns; takes dictionary of column
          names mapped to desired dtype
        :param numeric_dtype: if not None, changes all numeric types into this type
        :param names: specify new names for columns; takes dictionary of column
          index mapped to desired name
        :param ind: first column index (in case DataFrame are concatenated)
        :param tool: `'pandas'` or `'nimbusml'`
        :return: schema as a string

        """
        if isinstance(options.get('dtype', None), dict) and \
                options.get('numeric_dtype', None):
            for k, v in options['dtype'].items():
                if DataColumn(name=k, type=v).is_numeric():
                    raise ValueError(
                        "dtype and numeric_dtype contradict. If "
                        "numeric_dtype is specified, dtype can only change "
                        "a column into a non numeric type ({0})".format(
                            options['dtype']))

        header = options.pop('header', True)
        ind = options.pop('ind', 0)
        tool = options.pop('tool', 'pandas')

        def enumerate_blocks(blocks):
            start = blocks[0]
            length = 1
            for b in blocks[1:]:
                if b == start + length:
                    length += 1
                else:
                    yield (start, length)
                    start = b
                    length = 1
            yield (start, length)

        def clean_name(col):
            if isinstance(col, tuple):
                # Multi-Index --> converted into a string.
                return '.'.join(clean_name(_) for _ in col)
            if isinstance(col, (str, text_type)):
                return col.strip().replace(" ", "").replace(':', '')
            elif isinstance(col, tuple):
                # multilevel index
                return col
            elif isinstance(col, six.integer_types):
                # reads a file with no header
                return "c%d" % col
            else:
                raise TypeError(
                    "col must be str or tuple not {0}".format(
                        type(col)))

        cont = [c for c in data if c is not None]

        if len(cont) == 1:
            X = cont[0]

            if isinstance(X, (ndarray, csr_matrix)):
                length = X.shape[1]
                sch = [
                    DataColumn(
                        name='Data',
                        pos=ind,
                        type=X.dtype,
                        length=length)]
            elif isinstance(X, Series):
                sch = [DataColumn(type=X.dtype, pos=ind, name=X.name)]
            elif isinstance(X, DataFrame):
                if options.get('collapse', False):
                    collapse = options['collapse']
                    if collapse not in (False, True, 'all'):
                        raise TypeError("collapse must be a boolean or 'all'")
                    if options.get('numeric_dtype'):
                        # We must change the type before but it should
                        # also affect the block.
                        dtype = options.get('numeric_dtype')
                        for c in X.columns:
                            td = DataColumn(name='_', type=X[c].dtype)
                            if td.is_numeric():
                                X[c] = X[c].astype(dtype)
                        # Merging blocks
                        if collapse is True:
                            names = list(X.columns)
                            dtype = dict(zip(names, X.dtypes))
                            st = StringIO()
                            X.to_csv(st, index=False, sep="\t", header=False)
                            st = StringIO(st.getvalue())
                            X = read_csv(
                                st,
                                sep="\t",
                                names=names,
                                dtype=dtype,
                                header=None,
                                error_bad_lines=False,
                                warn_bad_lines=False)

                    names = options.get('names', None)
                    if isinstance(names, dict):
                        names = set(_ for _ in names if isinstance(_, six.integer_types))
                    elif isinstance(names, list):
                        names = set(range(len(names)))
                    else:
                        names = set()

                    # breaking blocs
                    if collapse is True:
                        for n in names:
                            c = X.columns[n]
                            dt = X[c].dtype
                            X[c] = X[c].astype(str)
                            X[c] = X[c].astype(dt)

                        # merging columns
                        sch = []
                        cols = X.columns
                        res = []
                        pos = 0
                        for b in X._data.blocks:
                            row = eval(str(list(b._mgr_locs)))
                            for bl in enumerate_blocks(row):
                                n = cols[bl[0]]
                                if n.endswith("_0"):
                                    n = n[:-2]
                                sch.append(
                                    DataColumn(
                                        type=b.dtype,
                                        pos=bl[0] + ind,
                                        length=bl[1],
                                        name=n))
                            pos += b.shape[0]
                    elif collapse == 'all':
                        sch = []
                        keep = {}
                        for pos, (col, dtype) in enumerate(
                                zip(X.columns, X.dtypes)):
                            if pos in names:
                                sch.append(
                                    DataColumn(
                                        type=dtype,
                                        pos=pos,
                                        name=col))
                            else:
                                sdt = str(dtype)
                                if sdt not in keep:
                                    keep[sdt] = col, []
                                keep[sdt][1].append(pos + ind)
                        for k, v in sorted(keep.items()):
                            sch.append(
                                DataColumn(
                                    type=k, pos=v[1], length=len(
                                        v[1]), name=clean_name(
                                        v[0])))
                else:
                    sch = [DataColumn(type=X[c].dtype, pos=i + ind,
                                      name=clean_name(c), clean_name=True)
                           for i, c in enumerate(X.columns)]
            elif hasattr(X, 'read') or isinstance(X, str) or (
                    six.PY2 and isinstance(X, (str, text_type))):
                # A stream like StringIO.
                res = DataSchema.read_schema_file(
                    X, tool=tool, header=header, **options)
                return res
            elif isinstance(X, list):
                ser = DataFrame(X[:10] if len(X) > 10 else X)
                res = DataSchema.read_schema(ser, **options)
                return res
            else:
                raise TypeError(
                    "Unable to guess the schema for type '{0}'".format(
                        type(X)))
            final_schema = sch

        elif len(cont) > 1:
            ind = 0
            final_schema = None
            for i, X in enumerate(cont):
                sch = DataSchema.read_schema(X, ind=ind, features=i == 0)
                ind += len(sch)
                if final_schema is None:
                    final_schema = list(sch.columns.values())
                else:
                    final_schema.extend(sch.columns.values())
        else:
            raise RuntimeError("Data is missing, no schema can be returned.")

        if options.get('numeric_dtype'):
            numeric_dtype = options['numeric_dtype']
            final_schema = [
                c.change_type(numeric_dtype) if c.is_numeric() else c for c in
                final_schema]

        opt = dict(header=True if header else False)
        opt.update({k: v for k, v in options.items()
                    if k in {'header', 'sep'}})
        final_schema.sort()
        return DataSchema(final_schema, **opt)


class COL:
    """
    Column selector.
    """

    def __init__(self, expr, to=None, cont=None):
        """
        :param expr: input column
        :param to: output column, (overwrites input if not specified)
        :param cont: input container (used to check the input exists),
            if not specified (None), the container is assumed to be the
            previous transform in the pipeline
        """
        if not isinstance(expr, (str, list, six.integer_types)):
            raise TypeError(
                "expr must be a string, int or a list of string, int.".format(
                    expr))
        self.expr = expr
        self.to = to
        self.container = cont

    def __add__(self, col2):
        """
        Concatenates two set of columns.
        """
        if id(self.container) != id(col2.container):
            raise RuntimeError("Only one container is allowed.")
        if self.to is not None and col2.to is not None and self.to != col2.to:
            raise RuntimeError(
                "Conflicts for to: {0} != {1}".format(
                    self.to, col2.to))
        to = self.to or col2.to

        res = []
        if isinstance(self.expr, list):
            res.extend(self.expr)
        else:
            res.append(self.expr)

        if isinstance(col2.expr, list):
            res.extend(col2.expr)
        else:
            res.append(col2.expr)

        return COL(res, to=to, cont=self.container)

    def get_in(self, container=None):
        """
        Returns the list of selected columns.

        :param container: overwrites the container specifed in the constructor
        """
        container = container or self.container
        set_expr = set('.*?[\\{')

        def get_cols(expr):
            if ',' in expr:
                spl = expr.split(',')
                return [get_cols(_) for _ in spl]
            else:
                e = set(expr)
                if e & set_expr:
                    # regular expression, needs to container to get them.
                    raise NotImplementedError(
                        "Regular expression are not yet"
                        " implemented '{0}'.".format(
                            expr))
                else:
                    return [expr]

        if isinstance(self.expr, str):
            if ':' in self.expr:
                if container is None:
                    raise ValueError(
                        'A slice cannot be defined without a container "\
                        "specified the constructor.')
                else:
                    raise NotImplementedError(
                        "Not yet implemented for expr='{0}'".format(
                            self.expr))
            else:
                cols = get_cols(self.expr)
        elif isinstance(self.expr, six.integer_types):
            cols = [str(self.expr)]
        elif isinstance(self.expr, list):
            cols = []
            for e in self.expr:
                cols.extend(get_cols(e))
        return cols

    def get_out(self):
        """
        Returns the outputted column.
        """
        if self.to:
            if isinstance(self.to, str):
                return [self.to]
            else:
                return self.to
        else:
            return None
