# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
# - Generated by tools/entrypoint_compiler.py: do not edit by hand
"""
ClassifierBestDiverseSelector
"""

__all__ = ["ClassifierBestDiverseSelector"]


from ...internal.core.ensemble.sub_model_selector._classifierbestdiverseselector import \
    ClassifierBestDiverseSelector as core
from ...internal.utils.utils import trace
from .diversity_measure import ClassifierDisagreement


class ClassifierBestDiverseSelector(core):
    """
    **Description**
        Combines the models whose predictions are as diverse as possible.

    :param diversity_metric_type: The metric type to be used to find the
        diversity among base learners.

    :param learners_selection_proportion: The proportion of best base learners
        to be selected. The range is 0.0-1.0.

    :param validation_dataset_proportion: The proportion of instances to be
        selected to test the individual base learner. If it is 0, it uses
        training set.

    :param params: Additional arguments sent to compute engine.

    """

    @trace
    def __init__(
            self,
            diversity_metric_type=ClassifierDisagreement(),
            learners_selection_proportion=0.5,
            validation_dataset_proportion=0.3,
            **params):
        core.__init__(
            self,
            diversity_metric_type=diversity_metric_type,
            learners_selection_proportion=learners_selection_proportion,
            validation_dataset_proportion=validation_dataset_proportion,
            **params)

    def get_params(self, deep=False):
        """
        Get the parameters for this operator.
        """
        return core.get_params(self)
