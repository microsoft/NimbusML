# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
# - Generated by tools/entrypoint_compiler.py: do not edit by hand
"""
RegressorAverage
"""

__all__ = ["RegressorAverage"]


from ....utils.entrypoints import Component
from ....utils.utils import trace


class RegressorAverage(Component):
    """
    **Description**
        Computes the average of the outputs of the trained models

    :param params: Additional arguments sent to compute engine.

    """

    @trace
    def __init__(
            self,
            **params):

        self.kind = 'EnsembleRegressionOutputCombiner'
        self.name = 'Average'
        self.settings = {}

        super(
            RegressorAverage,
            self).__init__(
            name=self.name,
            settings=self.settings,
            kind=self.kind)
