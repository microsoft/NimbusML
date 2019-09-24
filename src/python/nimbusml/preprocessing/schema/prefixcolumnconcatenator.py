# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
"""
PrefixColumnConcatenator
"""

__all__ = ["PrefixColumnConcatenator"]


from sklearn.base import TransformerMixin

from ...base_transform import BaseTransform
from ...internal.core.preprocessing.schema.prefixcolumnconcatenator import \
    PrefixColumnConcatenator as core
from ...internal.utils.utils import trace


class PrefixColumnConcatenator(core, BaseTransform, TransformerMixin):
    """

    Combines several columns into a single vector-valued column by prefix.

    .. remarks::
        ``PrefixColumnConcatenator`` creates a single vector-valued column from
        multiple
        columns. It can be performed on data before training a model. The
        concatenation
        can significantly speed up the processing of data when the number of
        columns
        is as large as hundreds to thousands.

    :param columns: a dictionary of key-value pairs, where key is the output
        column name and value is a list of input column names.

         * Only one key-value pair is allowed.
         * Input column type: numeric or string.
         * Output column type:
        `Vector Type </nimbusml/concepts/types#vectortype-column>`_.

        The << operator can be used to set this value (see
        `Column Operator </nimbusml/concepts/columns>`_)

        For example
         * ColumnConcatenator(columns={'features': ['age', 'parity',
        'induced']})
         * ColumnConcatenator() << {'features': ['age', 'parity',
        'induced']})

        For more details see `Columns </nimbusml/concepts/columns>`_.

    :param params: Additional arguments sent to compute engine.

    .. seealso::
        :py:class:`ColumnDropper
        <nimbusml.preprocessing.schema.ColumnDropper>`,
        :py:class:`ColumnSelector
        <nimbusml.preprocessing.schema.ColumnSelector>`.

    .. index:: transform, schema

    Example:
       .. literalinclude:: /../nimbusml/examples/PrefixColumnConcatenator.py
              :language: python
    """

    @trace
    def __init__(
            self,
            columns=None,
            **params):

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
