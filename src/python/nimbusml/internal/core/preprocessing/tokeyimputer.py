# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
# - Generated by tools/entrypoint_compiler.py: do not edit by hand
"""
ToKeyImputer
"""

__all__ = ["ToKeyImputer"]


from ...entrypoints.transforms_categoryimputer import \
    transforms_categoryimputer
from ...utils.utils import trace
from ..base_pipeline_item import BasePipelineItem, DefaultSignature


class ToKeyImputer(BasePipelineItem, DefaultSignature):
    """
    **Description**
        Turns the given column into a column of its string representation

    :param destination: Output column, if not specified defaults to the input.

    :param params: Additional arguments sent to compute engine.

    """

    @trace
    def __init__(
            self,
            destination=None,
            **params):
        BasePipelineItem.__init__(
            self, type='transform', **params)

        self.destination = destination

    @property
    def _entrypoint(self):
        return transforms_categoryimputer

    @trace
    def _get_node(self, **all_args):
        algo_args = dict(
            source=self.source,
            destination=self.destination)

        all_args.update(algo_args)
        return self._entrypoint(**all_args)
