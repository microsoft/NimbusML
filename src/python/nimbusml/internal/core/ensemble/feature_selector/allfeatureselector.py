# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
# - Generated by tools/entrypoint_compiler.py: do not edit by hand
"""
AllFeatureSelector
"""

__all__ = ["AllFeatureSelector"]


from ....utils.entrypoints import Component
from ....utils.utils import trace


class AllFeatureSelector(Component):
    """
    **Description**
        temp

    :param params: Additional arguments sent to compute engine.

    """

    @trace
    def __init__(
            self,
            **params):

        self.kind = 'EnsembleFeatureSelector'
        self.name = 'AllFeatureSelector'
        self.settings = {}

        super(
            AllFeatureSelector,
            self).__init__(
            name=self.name,
            settings=self.settings,
            kind=self.kind)
