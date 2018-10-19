# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
"""
Helpers to process schemas.
"""
from numpy import ndarray
from pandas import DataFrame, Series
from scipy.sparse import csr_matrix

from .data_roles import Role
from .data_schema import COL
from .data_stream import ViewDataStream, ViewBasePipelineItem


def _extract_label_column(learner, schema_y=None):
    if schema_y is not None:
        label_column = schema_y[0].Name
    elif learner.label_column is not None:
        label_column = learner.label_column
    else:
        label_column = Role.Label
    return label_column


def _extract_columns(columns, side, none_allowed):
    if columns is None:
        if none_allowed:
            return None
        raise ValueError(
            "'None' {0} passed when it cannot be none.".format(side))
    if isinstance(columns, Series):
        columns = [columns.name]
    elif isinstance(columns, DataFrame):
        columns = [_ for _ in columns.columns]
    elif isinstance(columns, str):
        columns = [columns]
    elif isinstance(columns, (ViewDataStream, ViewBasePipelineItem)):
        columns = columns.columns
    elif isinstance(columns, ndarray):
        # columns.flags['OWNDATA']
        raise ValueError(
            "There is not easy way to retrieve the original columns "
            "from a numpy view. Use COL.")
    elif isinstance(columns, csr_matrix):
        # columns.flags['OWNDATA']
        raise ValueError(
            "There is not easy way to retrieve the original columns from a "
            "csr_matrix view. Use COL.")
    elif isinstance(columns, COL):
        if side == 'input':
            columns = columns.get_in()
        elif side == 'output':
            columns = columns.get_out()
        else:
            raise ValueError(
                "side must be 'input' our 'output' not '{0}'".format(side))
    if not isinstance(columns, list):
        raise ValueError(
            "{0} has to be a list of strings, instead got {1}".format(
                side, type(columns)))
    return columns
