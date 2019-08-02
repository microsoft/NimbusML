# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
# - Generated by tools/entrypoint_compiler.py: do not edit by hand
"""
BootstrapSelector
"""

__all__ = ["BootstrapSelector"]


from ...internal.core.ensemble.subset_selector.bootstrapselector import \
    BootstrapSelector as core
from ...internal.utils.utils import trace


class BootstrapSelector(core):
    """
    **Description**
        temp

    :param feature_selector: The Feature selector.

    :param params: Additional arguments sent to compute engine.

    """

    @trace
    def __init__(
            self,
            feature_selector=None,
            **params):
        core.__init__(
            self,
            feature_selector=feature_selector,
            **params)

    def get_params(self, deep=False):
        """
        Get the parameters for this operator.
        """
        return core.get_params(self)
