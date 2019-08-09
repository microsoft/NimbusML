# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
# - Generated by tools/entrypoint_compiler.py: do not edit by hand
"""
ClassifierBestPerformanceSelector
"""

__all__ = ["ClassifierBestPerformanceSelector"]


from ...internal.core.ensemble.sub_model_selector._classifierbestperformanceselector import \
    ClassifierBestPerformanceSelector as core
from ...internal.utils.utils import trace


class ClassifierBestPerformanceSelector(core):
    """

    **Description**
    Combines only the models with the best performance.


    :param metric_name: the metric type to be used to find the weights for
        each model. Can be ``"AccuracyMicro"``, ``"AccuracyMacro"``,
        ``"LogLoss"``, or ``"LogLossReduction"``.

    :param learners_selection_proportion: The proportion of best base learners
        to be selected. The range is 0.0-1.0.

    :param validation_dataset_proportion: The proportion of instances to be
        selected to test the individual base learner. If it is 0, it uses
        training set.

    :param params: Additional arguments sent to compute engine.

    .. seealso::
        :py:class:`EnsembleClassifier
        <nimbusml.ensemble.EnsembleClassifier>`

        * Submodel selectors:
        :py:class:`ClassifierAllSelector
        <nimbusml.ensemble.sub_model_selector.ClassifierAllSelector>`,
        :py:class:`ClassifierBestDiverseSelector
        <nimbusml.ensemble.sub_model_selector.ClassifierBestDiverseSelector>`

        * Output combiners:
        :py:class:`ClassifierAverage
        <nimbusml.ensemble.output_combiner.ClassifierAverage>`,
        :py:class:`ClassifierMedian
        <nimbusml.ensemble.output_combiner.ClassifierMedian>`,
        :py:class:`ClassifierStacking
        <nimbusml.ensemble.output_combiner.ClassifierStacking>`,
        :py:class:`ClassifierVoting
        <nimbusml.ensemble.output_combiner.ClassifierVoting>`


    .. index:: models, ensemble, classification

    Example:
       .. literalinclude:: /../nimbusml/examples/EnsembleClassifier.py
              :language: python
    """

    @trace
    def __init__(
            self,
            metric_name='AccuracyMicro',
            learners_selection_proportion=0.5,
            validation_dataset_proportion=0.3,
            **params):
        core.__init__(
            self,
            metric_name=metric_name,
            learners_selection_proportion=learners_selection_proportion,
            validation_dataset_proportion=validation_dataset_proportion,
            **params)

    def get_params(self, deep=False):
        """
        Get the parameters for this operator.
        """
        return core.get_params(self)
