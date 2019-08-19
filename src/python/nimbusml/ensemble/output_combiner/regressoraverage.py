# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
# - Generated by tools/entrypoint_compiler.py: do not edit by hand
"""
RegressorAverage
"""

__all__ = ["RegressorAverage"]


from ...internal.core.ensemble.output_combiner.regressoraverage import \
    RegressorAverage as core
from ...internal.utils.utils import trace


class RegressorAverage(core):
    """
    **Description**
        Computes the average of the outputs of the trained models

    :param params: Additional arguments sent to compute engine.

    """

    @trace
    def __init__(
            self,
            **params):
        core.__init__(
            self,
            **params)

    def get_params(self, deep=False):
        """
        Get the parameters for this operator.
        """
        return core.get_params(self)
