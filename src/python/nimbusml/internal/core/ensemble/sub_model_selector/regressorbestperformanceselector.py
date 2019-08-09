# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
# - Generated by tools/entrypoint_compiler.py: do not edit by hand
"""
RegressorBestPerformanceSelector
"""

__all__ = ["RegressorBestPerformanceSelector"]

import numbers

from ....utils.entrypoints import Component
from ....utils.utils import trace, try_set


class RegressorBestPerformanceSelector(Component):
    """

    **Description**
    Computes the weighted average of the outputs of the trained models


    :param metric_name: the metric type to be used to find the weights for
        each model. Can be ``"L1"``, ``"L2"``, ``"Rms"``, or ``"Loss"``, or
        ``"RSquared"``.

    :param learners_selection_proportion: The proportion of best base learners
        to be selected. The range is 0.0-1.0.

    :param validation_dataset_proportion: The proportion of instances to be
        selected to test the individual base learner. If it is 0, it uses
        training set.

    :param params: Additional arguments sent to compute engine.

    .. seealso::
        :py:class:`EnsembleRegressor
        <nimbusml.ensemble.EnsembleRegressor>`

        * Submodel selectors:
        :py:class:`RegressorAllSelector
        <nimbusml.ensemble.sub_model_selector.RegressorAllSelector>`,
        :py:class:`RegressorBestDiverseSelector
        <nimbusml.ensemble.sub_model_selector.RegressorBestDiverseSelector>`

        * Output combiners:
        :py:class:`RegressorAverage
        <nimbusml.ensemble.output_combiner.RegressorAverage>`,
        :py:class:`RegressorMedian
        <nimbusml.ensemble.output_combiner.RegressorMedian>`,
        :py:class:`RegressorStacking
        <nimbusml.ensemble.output_combiner.RegressorStacking>`


    .. index:: models, ensemble, classification

    Example:
       .. literalinclude:: /../nimbusml/examples/EnsembleClassifier.py
              :language: python
    """

    @trace
    def __init__(
            self,
            metric_name='L1',
            learners_selection_proportion=0.5,
            validation_dataset_proportion=0.3,
            **params):

        self.metric_name = metric_name
        self.learners_selection_proportion = learners_selection_proportion
        self.validation_dataset_proportion = validation_dataset_proportion
        self.kind = 'EnsembleRegressionSubModelSelector'
        self.name = 'BestPerformanceRegressionSelector'
        self.settings = {}

        if metric_name is not None:
            self.settings['MetricName'] = try_set(
                obj=metric_name, none_acceptable=True, is_of_type=str, values=[
                    'L1', 'L2', 'Rms', 'Loss', 'RSquared'])
        if learners_selection_proportion is not None:
            self.settings['LearnersSelectionProportion'] = try_set(
                obj=learners_selection_proportion,
                none_acceptable=True,
                is_of_type=numbers.Real)
        if validation_dataset_proportion is not None:
            self.settings['ValidationDatasetProportion'] = try_set(
                obj=validation_dataset_proportion,
                none_acceptable=True,
                is_of_type=numbers.Real)

        super(
            RegressorBestPerformanceSelector,
            self).__init__(
            name=self.name,
            settings=self.settings,
            kind=self.kind)
