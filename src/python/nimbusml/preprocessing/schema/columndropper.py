# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
# - Generated manually! Required for backward compatibility.
"""
ColumnDropper
"""

__all__ = ["ColumnDropper"]


import warnings
from sklearn.base import TransformerMixin

from ...base_transform import BaseTransform
from ...internal.core.preprocessing.schema.columndropper import \
    ColumnDropper as core
from ...internal.utils.utils import trace


class ColumnDropper(core, BaseTransform, TransformerMixin):
    """

    Specified columns to drop from the dataset.

    :param columns: a list of strings representing the column names to
        perform the transformation on.

        The << operator can be used to set this value (see
        `Column Operator </nimbusml/concepts/columns>`_)

        For example
         * ColumnDropper(columns=['education', 'age'])
         * ColumnDropper() << ['education', 'age']

        For more details see `Columns </nimbusml/concepts/columns>`_.

    :param params: Additional arguments sent to compute engine.

    .. seealso::
        :py:class:`ColumnConcatenator
        <nimbusml.preprocessing.schema.ColumnConcatenator>`,
        :py:class:`ColumnSelector
        <nimbusml.preprocessing.schema.ColumnSelector>`.

    .. index:: transform, schema

    Example:
       .. literalinclude:: /../nimbusml/examples/ColumnDropper.py
              :language: python
    """

    @trace
    def __init__(
            self,
            columns=None,
            **params):
        
        warnings.warn(
            "ColumnDropper is will be deprecated in future releases."
            "Use ColumnSelector(drop_columns) instead.", 
            PendingDeprecationWarning
        )

        if columns:
            params['columns'] = columns
        BaseTransform.__init__(self, **params)
        core.__init__(
            self,
            **params)
        self._columns = columns

    def get_params(self, deep=False):
        """
        Get the parameters for this operator.
        """
        return core.get_params(self)
