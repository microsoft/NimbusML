# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
# - Generated manually! Required for backward compatibility.
"""
ColumnDropper
"""

__all__ = ["ColumnDropper"]


from ....entrypoints.transforms_columnselector import transforms_columnselector
from ....utils.utils import trace
from ...base_pipeline_item import BasePipelineItem, NoOutputSignature


class ColumnDropper(BasePipelineItem, NoOutputSignature):
    """

    Specified columns to drop from the dataset.

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
            **params):
        BasePipelineItem.__init__(
            self, type='transform', **params)

    @property
    def _entrypoint(self):
        return transforms_columnselector

    @trace
    def _get_node(self, **all_args):

        input_columns = self.input
        if input_columns is None and 'input' in all_args:
            input_columns = all_args['input']
        if 'input' in all_args:
            all_args.pop('input')

        # validate input
        if input_columns is None:
            raise ValueError(
                "'None' input passed when it cannot be none.")

        if not isinstance(input_columns, list):
            raise ValueError(
                "input has to be a list of strings, instead got %s" %
                type(input_columns))

        algo_args = dict(
            column=input_columns,
            keep_columns=None,
            drop_columns=input_columns,
            keep_hidden=False,
            ignore_missing=False)

        all_args.update(algo_args)
        return self._entrypoint(**all_args)
