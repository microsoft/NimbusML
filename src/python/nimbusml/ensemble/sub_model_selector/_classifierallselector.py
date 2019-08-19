# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
# - Generated by tools/entrypoint_compiler.py: do not edit by hand
"""
ClassifierAllSelector
"""

__all__ = ["ClassifierAllSelector"]


from ...internal.core.ensemble.sub_model_selector._classifierallselector import \
    ClassifierAllSelector as core
from ...internal.utils.utils import trace


class ClassifierAllSelector(core):
    """
    **Description**
        Combines all the models to create the output. This is the default submodel selector.

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
