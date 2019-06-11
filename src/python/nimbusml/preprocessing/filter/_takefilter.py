# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
# - Generated by tools/entrypoint_compiler.py: do not edit by hand
"""
TakeFilter
"""

__all__ = ["TakeFilter"]


from sklearn.base import TransformerMixin

from ...base_transform import BaseTransform
from ...internal.core.preprocessing.filter._takefilter import \
    TakeFilter as core
from ...internal.utils.utils import trace


class TakeFilter(core, BaseTransform, TransformerMixin):
    """

    Take N first rows of the dataset, allowing limiting input to a subset
    of rows.

    :param columns: a string representing the column name to perform the
        transformation on.

        * Input column type: numeric.
        * Output column type: numeric.

        The << operator can be used to set this value (see
        `Column Operator </nimbusml/concepts/columns>`_)

        For example
         * TakeFilter(columns='age')
         * TakeFilter() << {'age'}

        For more details see `Columns </nimbusml/concepts/columns>`_.

    :param count: number of rows to keep from the beginning of the dataset.

    :param params: Additional arguments sent to compute engine.

    .. index:: transform, random

    Example:
       .. literalinclude:: /../nimbusml/examples/SkipFilter.py
              :language: python
    """

    @trace
    def __init__(
            self,
            count,
            columns=None,
            **params):

        if columns:
            params['columns'] = columns
        BaseTransform.__init__(self, **params)
        core.__init__(
            self,
            count=count,
            **params)
        self._columns = columns

    def get_params(self, deep=False):
        """
        Get the parameters for this operator.
        """
        return core.get_params(self)
