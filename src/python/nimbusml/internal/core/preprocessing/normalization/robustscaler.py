# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
# - Generated by tools/entrypoint_compiler.py: do not edit by hand
"""
RobustScaler
"""

__all__ = ["RobustScaler"]


from ....entrypoints.transforms_robustscalar import transforms_robustscalar
from ....utils.utils import trace
from ...base_pipeline_item import BasePipelineItem, DefaultSignature


class RobustScaler(BasePipelineItem, DefaultSignature):
    """
    **Description**
        Removes the median and scales the data according to the quantile range.

    :param center: If True, center the data before scaling.

    :param scale: If True, scale the data to interquartile range.

    :param quantile_min: Min for the quantile range used to calculate scale.

    :param quantile_max: Max for the quantile range used to calculate scale.

    :param params: Additional arguments sent to compute engine.

    """

    @trace
    def __init__(
            self,
            center=True,
            scale=True,
            quantile_min=25.0,
            quantile_max=75.0,
            **params):
        BasePipelineItem.__init__(
            self, type='transform', **params)

        self.center = center
        self.scale = scale
        self.quantile_min = quantile_min
        self.quantile_max = quantile_max

    @property
    def _entrypoint(self):
        return transforms_robustscalar

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
            output_columns = input_columns

        if not isinstance(output_columns, list):
            raise ValueError(
                "output has to be a list of strings, instead got %s" %
                type(output_columns))

        algo_args = dict(
            column=[
                dict(
                    Source=i,
                    Name=o) for i,
                o in zip(
                    input_columns,
                    output_columns)] if input_columns else None,
            center=self.center,
            scale=self.scale,
            quantile_min=self.quantile_min,
            quantile_max=self.quantile_max)

        all_args.update(algo_args)
        return self._entrypoint(**all_args)
