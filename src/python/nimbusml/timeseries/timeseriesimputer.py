# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
# - Generated by tools/entrypoint_compiler.py: do not edit by hand
"""
TimeSeriesImputer
"""

__all__ = ["TimeSeriesImputer"]


from sklearn.base import TransformerMixin

from ..base_transform import BaseTransform
from ..internal.core.timeseries.timeseriesimputer import \
    TimeSeriesImputer as core
from ..internal.utils.utils import trace


class TimeSeriesImputer(core, BaseTransform, TransformerMixin):
    """
    **Description**
        Fills in missing row and values

    :param columns: see `Columns </nimbusml/concepts/columns>`_.

    :param time_series_column: Column representing the time.

    :param grain_columns: List of grain columns.

    :param filter_columns: Columns to filter.

    :param filter_mode: Filter mode. Either include or exclude.

    :param impute_mode: Mode for imputing, defaults to ForwardFill if not
        provided.

    :param supress_type_errors: Supress the errors that would occur if a column
        and impute mode are imcompatible. If true, will skip the column. If
        false, will stop and throw an error.

    :param params: Additional arguments sent to compute engine.

    """

    @trace
    def __init__(
            self,
            time_series_column,
            grain_columns=None,
            filter_columns=None,
            filter_mode='Exclude',
            impute_mode='ForwardFill',
            supress_type_errors=False,
            columns=None,
            **params):

        if columns:
            params['columns'] = columns
        BaseTransform.__init__(self, **params)
        core.__init__(
            self,
            time_series_column=time_series_column,
            grain_columns=grain_columns,
            filter_columns=filter_columns,
            filter_mode=filter_mode,
            impute_mode=impute_mode,
            supress_type_errors=supress_type_errors,
            **params)
        self._columns = columns

    def get_params(self, deep=False):
        """
        Get the parameters for this operator.
        """
        return core.get_params(self)
