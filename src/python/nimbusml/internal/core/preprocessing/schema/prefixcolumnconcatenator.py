# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
"""
PrefixColumnConcatenator
"""

__all__ = ["PrefixColumnConcatenator"]


from ....entrypoints.transforms_prefixcolumnconcatenator import \
    transforms_prefixcolumnconcatenator
from ....utils.utils import trace
from ...base_pipeline_item import BasePipelineItem, DefaultSignature


class PrefixColumnConcatenator(BasePipelineItem, DefaultSignature):
    """

    Combines several columns into a single vector-valued column by prefix

    .. remarks::
        ``PrefixColumnConcatenator`` creates a single vector-valued column from
        multiple
        columns. It can be performed on data before training a model. The
        concatenation
        can significantly speed up the processing of data when the number of
        columns
        is as large as hundreds to thousands.

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
            **params):
        BasePipelineItem.__init__(
            self, type='transform', **params)

    @property
    def _entrypoint(self):
        return transforms_prefixcolumnconcatenator

    @trace
    def _get_node(self, **all_args):

        input_columns = self.input
        if input_columns is None and 'input' in all_args:
            input_columns = all_args['input']
        if 'input' in all_args:
            all_args.pop('input')

        output_columns = self.output
        if output_columns is None and 'output' in all_args:
            output_columns = all_args['output']
        if 'output' in all_args:
            all_args.pop('output')

        # validate input
        if input_columns is None:
            raise ValueError(
                "'None' input passed when it cannot be none.")

        if not isinstance(input_columns, list):
            raise ValueError(
                "input has to be a list of strings, instead got %s" %
                type(input_columns))

        # validate output
        if output_columns is None:
            raise ValueError(
                "'None' output passed when it cannot be none.")

        if not isinstance(output_columns, list):
            raise ValueError(
                "output has to be a list of strings, instead got %s" %
                type(output_columns))

        algo_args = dict(
            column=[
                dict(
                    Source=i, Name=o) for i, o in zip(
                    input_columns, output_columns)] if input_columns else None)

        all_args.update(algo_args)
        return self._entrypoint(**all_args)
