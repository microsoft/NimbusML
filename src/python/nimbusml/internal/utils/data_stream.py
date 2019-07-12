# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
"""
Owns nimbusml's containers.
"""
import os
import tempfile
from shutil import copyfile

from .data_roles import DataRoles
from .data_schema import DataSchema
from .utils import trace


class DataStream(object):
    """
    Defines a streaming container (a data view).
    """
    # REVIEW: this list needs to be updated every time a new learner invents
    # a new role.
    # Maybe every learner and transform could declare the roles it requires to
    # train and predict.
    _allowed = DataRoles._allowed

    def __init__(self, schema, roles=None):
        """
        :param schema: data schema
        :param roles: roles definition
        """
        if schema is None:
            raise ValueError("The schema must be specified.")
        if isinstance(schema, str):
            schema = DataSchema(schema)
        if not isinstance(schema, DataSchema):
            raise TypeError(
                "schema must be of DataSchema (not {0}).".format(
                    type(schema)))
        if roles is not None:
            if not isinstance(roles, DataRoles):
                raise TypeError(
                    "roles must be a dictionary (not {0}).".format(
                        type(roles)))
        self._schema = schema
        self._roles = roles if roles else DataRoles()
        self._verify_roles()

    def clone(self):
        """
        Copy/clone the object.
        """
        if not isinstance(self, DataStream):
            raise NotImplementedError(
                "Method clone was not overwritten for class '{0}'".format(
                    type(self)))
        return DataStream(self._schema.clone(), self._roles.clone())

    def _verify_roles(self):
        """
        Checks that a role belongs to the schema.
        """
        self._roles._verify_roles(self._schema)

    @property
    def schema(self):
        """
        Return the :py:func:`DataSchema <nimbusml.DataSchema>` of
        the FileDataStream.
        """
        return self._schema

    @property
    def roles(self):
        return self._roles

    def _set_role(self, role_name, column_name):
        self._roles._set_role(role_name, column_name)

    def _get_role(self, role_name):
        return self._roles._get_role(role_name)

    def _has_role(self, role_name):
        return self._roles._has_role(role_name)

    def __repr__(self):
        return "DataStream('{0}',\n    {1})".format(self._schema, self._roles)

    def _pipe_data(self, n=-1, skip=0, columns=None, collect=True):
        from ... import Pipeline
        from ...preprocessing.filter import TakeFilter, SkipFilter
        from ...preprocessing.schema import ColumnSelector
        if columns is None:
            columns = [c.name for c in self.schema]
        steps = []
        if skip > 0:
            steps.append(SkipFilter(skip, columns=columns))
        if n > 0:
            steps.append(TakeFilter(n, columns=columns))
        if len(steps) == 0:
            steps.append(ColumnSelector(columns=columns))
        pipe = Pipeline(steps)

        if collect:
            return pipe.fit_transform(
                self, verbose=0, as_binary_data_stream=False)
        else:
            pipe.fit(self, verbose=0)
            return pipe

    def head(self, n=5, skip=0, columns=None, collect=True):
        """
        Returns the first rows. The method
        runs a pipeline on the data with a simple
        transform which skips *skip* rows and takes
        the following *n* rows. If *collect* is True,
        the data is returned as a dataframe, otherwise
        it returns a fitted pipeline to run
        (method *predict*).

        :param n: number of rows to keep
        :param skip: number of rows to skip
        :param columns: selects a subset of columns or None for all
        :param collect: returns either a pipeline (False) or the data in a
            dataframe (True)
        :returns: *DataFrame* or *Pipeline* (see parameter *collect*)
        """
        # Do not move these imports or the module fails
        # due to circular references.
        if n <= 0 and skip <= 0:
            raise ValueError("n or skip must be > 0")
        return self._pipe_data(
            n=n,
            skip=skip,
            columns=columns,
            collect=collect)

    def to_df(self, columns=None, collect=True):
        """
        Returns the data as a dataframe. The method
        runs a pipeline on the data and returns
        the outcome of method *predict*.
        If the data is big, less columns should be selected
        or less rows (with method *head*).

        :param collect: returns either a pipeline (False) or the data in a
            dataframe (True)
        :param columns: selects a subset of columns or None for all
        :returns: *DataFrame* or *Pipeline* (see parameter *collect*)
        """
        return self._pipe_data(n=-1, skip=0, columns=columns, collect=collect)


# REVIEW: Since BinaryDataStream is also based on a file, should we rename
# to TextDataStream?
# This would require updating the documentation, and possibly having the
# FileDataStream name as
# an alias until we eventually depracate it.


class FileDataStream(DataStream):
    """

    Data view from a file.

    .. remarks::
        FileDataStream enables training from files by streaming the
        examples sequentially. Some trainers require the
        full data matrix to be resident in memory, and will cache the
        data if required. For trainers that implement
        online or batch techniques, using FileDataStream will substantially
        reduce overall memory utilization. Runtime
        efficiency is also increased and data copying is minimized for
        ``nimbusml`` trainers/transforms when used in
        conjunction with FileDataStream text reader.

        A schema of the data is required to describe the column names,
        positions, types and delimiters. This can be
        provided explicitly to FileDataStream by using the
        :py:func:`DataSchema <nimbusml.DataSchema>` class
        to construct it, or optionally the
        :py:func:`FileDataStream.read_csv <nimbusml.FileDataStream.read_csv>`
        method can be used to infer the schema automatically. For more
        control over column names and index ranges, especially
        `Vector Type </nimbusml/concepts/types#vectortype-column>`_
        columns, the schema can be designed
        manually.

        For  more details of the schema format, refer to
        `Schema </nimbusml/concepts/schema#dataschema-class>`_
        and :py:func:`DataSchema <nimbusml.DataSchema>`.

    .. seealso::
        :py:func:`DataSchema <nimbusml.DataSchema>`.

    Example:
        .. code-block:: python

            from nimbusml import FileDataStream
            from nimbusml import Pipeline
            from nimbusml.ensemble import LightGbmRegressor
            from nimbusml.feature_extraction.categorical import OneHotVectorizer
            import numpy as np
            import pandas as pd

            data = pd.DataFrame(dict(real = [0.1, 2.2],
                                     text = ['word','class'],
                                     y = [1,3]))
            data.to_csv('data.csv', index = False, header = True)

            ds = FileDataStream.read_csv('data.csv', collapse = False,
                                        numeric_dtype = np.float32, sep = ',')
            ds.head()
            #   real   text    y
            #0   0.1   word  1.0
            #1   2.2  class  3.0
            exp = Pipeline([
                         OneHotVectorizer(columns = ['text']),
                         LightGbmRegressor(minimum_example_count_per_leaf = 1)
                        ])

            exp.fit(ds, 'y')

    """

    def __init__(self, filename, schema, roles=None):
        """
        :param filename: filename of a datasets
        :param schema: filename schema
        """
        super(FileDataStream, self).__init__(schema, roles)
        self._filename = filename

    def __repr__(self):
        return "FileDataStream('{2}',\n    '{0}',\n    {1})".format(
            self._schema, self._roles, self._filename.replace('\\', '\\\\'))

    def clone(self):
        """
        Copy/clone the object.
        """
        if not isinstance(self, FileDataStream):
            raise NotImplementedError(
                "Method clone was not overwritten for class '{0}'".format(
                    type(self)))
        return FileDataStream(
            self._filename,
            self._schema.clone(),
            self._roles.clone())

    def __getitem__(self, columns):
        """
        Creates a view on this stream.
        """
        if not isinstance(self, FileDataStream):
            raise TypeError(
                "Cannot create a view on type {0}.".format(
                    type(self)))
        return ViewDataStream(self, columns)

    @property
    def filename(self):
        return self._filename

    @staticmethod
    @trace
    def read_csv(filepath_or_buffer, tool=None, nrows=100, **kwargs):
        """
        Creates a *FileDataStream* from a filename or a buffer. For more
        details of the schema format for
        a FileDataStream, refer to
        `Schema </nimbusml/concepts/schema#dataschema-class>`_
        all the arguments that ``DataSchema.read_schema()`` uses applies to
        this method as well.

        :param filepath_or_buffer: filename or stream
        :param tool: parser to choose to guess the schema,
            this module ``'internal'`` or ``'pandas'``, if None,
            the function chooses the most relevant one given the
            additional arguments given to the function
        :param nrows: number of rows used to guess the schema
        :param numeric_dtype: changes all numeric types into the same one,
            recommended to use numpy.float32 in many cases
        :param collapse: (False by default), collapse columns for of the same
            type if it follows
            *read_csv* function. Use internal structure of a dataframe.
            If ``collapse* == 'all'``, the method collapses all
            columns not specified in parameter *names*.
        :param sep: seperation of the data columns, such as ',', or '/t'
        :param header: if the input data has a header, can be True or False
        :param names: rename the data columns, users can specify a dictionary
            with column number as the key, such as
            {0:'Label', 1:'GroupId', (2,None):'Features'}
            It renames columns 0, 1, as *Label* and *GroupId*.
            It renames columns 2:end with Features_0, ..., Features_2040.
        :param dtype: overwrite the data column types, users can specify a
            dictionary with column name as the key, such as
            {'column1':numpy.float32}
        :param kwargs: additional parameters sent to *read_csv*
            or the internal parser.
        :return: a FileDataStream instance
        """
        if tool is None:
            tool = "pandas" if 'schema' not in kwargs else 'internal'

        if tool == 'pandas':
            return FileDataStream.read_csv_pandas(
                filepath_or_buffer, nrows=nrows, **kwargs)
        elif tool == 'internal':
            if 'schema' not in kwargs:
                raise ValueError(
                    "Parameter schema is not defined. Use tool='pandas'.")
            return FileDataStream(filepath_or_buffer, kwargs['schema'])
        else:
            raise ValueError("Unknown tool '{0}'.".format(tool))

    @staticmethod
    @trace
    def read_csv_pandas(
            filepath_or_buffer,
            nrows=100,
            collapse=False,
            numeric_dtype=None,
            **kwargs):
        """
        Creates a *FileDataStream* from a filename or a buffer.

        :param filepath_or_buffer: filename or stream
        :param nrows: number of rows used to guess the schema
        :param kwargs: additional parameters sent to *read_csv* or the internal
        :param numeric_dtype: changes all numeric types into the same one
        :param collapse: collapse into one vector column all columns sharing
            the same type
        :return: a FileDataStream instance

        The method leverages
        `read_csv
        <https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html>`_
        to guess the schema of a filename with the first *nrows* of a file.
        """
        schema = DataSchema.read_schema(filepath_or_buffer, collapse=collapse,
                                        numeric_dtype=numeric_dtype, **kwargs)
        return FileDataStream(filepath_or_buffer, schema)


class ViewDataStream:
    """
    A view on an existing stream.
    Only selected columns can be accessed.
    """

    def __init__(self, parent, columns):
        """
        :param param: DataStream
        :param columns: column or list of selected columns
        """
        if not isinstance(parent, DataStream):
            raise TypeError(
                "parent must be a DataStream, not {0}".format(
                    type(parent)))
        self.parent = parent
        if isinstance(columns, str):
            self.columns = [columns]
        elif isinstance(columns, list):
            self.columns = columns
        else:
            raise TypeError(
                'columns must be a string or a list or string not {0}'.format(
                    type(columns)))

    def __getitem__(self, columns):
        """
        Creates a view on the parent of this stream.
        """
        return self.parent[columns]

    @property
    def schema(self):
        """Returns parent's schema."""
        return self.parent.schema


class ViewBasePipelineItem:
    """
    View on a BasePipelineItem.
    """

    def __init__(self, parent, columns):
        self.parent = parent
        if isinstance(columns, str):
            self.columns = [columns]
        elif isinstance(columns, list):
            self.columns = columns
        else:
            raise TypeError(
                'columns must be a string or a list or string not {0}'.format(
                    type(columns)))


class BinaryDataStream(DataStream):
    """
    Defines a data view.
    """

    def __init__(self, filename):
        # REVIEW: would be good to figure out a way to know the schema of the
        # binary IDV.
        super(BinaryDataStream, self).__init__(DataSchema(""))
        self._filename = filename

    def __repr__(self):
        return "BinaryDataStream('{2}',\n    '{0}',\n    {1})".format(
            self._schema, self._roles, self._filename.replace('\\', '\\\\'))

    def save(self, file):
        copyfile(self._filename, file)

    def to_df(self):
        # Do not move these imports or the module fails
        # due to circular references.
        from ..entrypoints.transforms_nooperation import transforms_nooperation
        from .entrypoints import Graph

        no_op = transforms_nooperation(
            data='$data', output_data='$output_data')
        graph_nodes = [no_op]
        graph = Graph(
            dict(
                data=''), dict(
                output_data=''), False, *(graph_nodes))
        (out_model, out_data, out_metrics) = graph.run(verbose=True, X=self)
        return out_data

    def head(self, n=5, skip=0):
        # Do not move these imports or the module fails
        # due to circular references.
        from ..entrypoints.transforms_rowtakefilter import \
            transforms_rowtakefilter
        from ..entrypoints.transforms_rowskipfilter import \
            transforms_rowskipfilter
        from .entrypoints import Graph
        if n == 0:
            raise ValueError("n must be > 0")
        graph_nodes = []
        if skip > 0:
            graph_nodes.append(
                transforms_rowskipfilter(
                    data='$data',
                    output_data='$output_skip',
                    count=skip))
        graph_nodes.append(
            transforms_rowtakefilter(
                data='$output_skip' if skip > 0 else '$data',
                output_data='$output_data',
                count=n))
        graph = Graph(
            dict(
                data=''), dict(
                output_data=''), False, *(graph_nodes))
        (out_model, out_data, out_metrics) = graph.run(verbose=True, X=self)
        return out_data

    def clone(self):
        """
        Copy/clone the object.
        """
        if not isinstance(self, BinaryDataStream):
            raise NotImplementedError(
                "Method clone was not overwritten for class '{0}'".format(
                    type(self)))
        return BinaryDataStream(self._filename)


class DprepDataStream(BinaryDataStream):
    """
    Defines a data view over dprep file.
    """

    def __init__(self, dataflow=None, filename=None):
        if dataflow is None and filename is None:
            raise ValueError('Both dataflow object and filename are None')
        super(DprepDataStream, self).__init__(DataSchema(""))
        if dataflow is not None:
            (fd, filename) = tempfile.mkstemp(suffix='.dprep')
            fl = os.fdopen(fd, "wt")
            fl.write(dataflow.to_json())
            fl.close()
        self._filename = filename

    def __repr__(self):
        return "DprepDataStream('{2}',\n    '{0}',\n    {1})".format(
            self._schema, self._roles, self._filename.replace('\\', '\\\\'))

    def clone(self):
        """
        Copy/clone the object.
        """
        if not isinstance(self, DprepDataStream):
            raise NotImplementedError(
                "Method clone was not overwritten for class '{0}'".format(
                    type(self)))
        return DprepDataStream(self._filename)