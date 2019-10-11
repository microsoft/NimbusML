# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
# - Generated by tools/entrypoint_compiler.py: do not edit by hand
"""
ClassifierWeightedAverage
"""

__all__ = ["ClassifierWeightedAverage"]


from ....utils.entrypoints import Component
from ....utils.utils import trace, try_set


class ClassifierWeightedAverage(Component):
    """

    **Description**
    Computes the weighted average of the outputs of the trained models


    :param weightage_name: the metric type to be used to find the weights for
        each model. Can be ``"AccuracyMicroAvg"`` or ``"AccuracyMacroAvg"``.

    :param normalize: Specifies the type of automatic normalization used:

        * ``"Auto"``: if normalization is needed, it is performed
          automatically. This is the default choice.
        * ``"No"``: no normalization is performed.
        * ``"Yes"``: normalization is performed.
        * ``"Warn"``: if normalization is needed, a warning
          message is displayed, but normalization is not performed.

        Normalization rescales disparate data ranges to a standard scale.
        Feature
        scaling ensures the distances between data points are proportional
        and
        enables various optimization methods such as gradient descent to
        converge
        much faster. If normalization is performed, a ``MinMax`` normalizer
        is
        used. It normalizes values in an interval [a, b] where ``-1 <= a <=
        0``
        and ``0 <= b <= 1`` and ``b - a = 1``. This normalizer preserves
        sparsity by mapping zero to zero.

    :param params: Additional arguments sent to compute engine.

    .. seealso::
        :py:class:`EnsembleClassifier
        <nimbusml.ensemble.EnsembleClassifier>`

        * Submodel selectors:
        :py:class:`ClassifierAllSelector
        <nimbusml.ensemble.sub_model_selector.ClassifierAllSelector>`,
        :py:class:`ClassifierBestDiverseSelector
        <nimbusml.ensemble.sub_model_selector.ClassifierBestDiverseSelector>`,
        :py:class:`ClassifierBestPerformanceSelector
        <nimbusml.ensemble.sub_model_selector.ClassifierBestPerformanceSelector>`

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
            weightage_name='AccuracyMicroAvg',
            normalize=True,
            **params):

        self.weightage_name = weightage_name
        self.normalize = normalize
        self.kind = 'EnsembleMulticlassOutputCombiner'
        self.name = 'MultiWeightedAverage'
        self.settings = {}

        if weightage_name is not None:
            self.settings['WeightageName'] = try_set(
                obj=weightage_name, none_acceptable=True, is_of_type=str, values=[
                    'AccuracyMicroAvg', 'AccuracyMacroAvg'])
        if normalize is not None:
            self.settings['Normalize'] = try_set(
                obj=normalize, none_acceptable=True, is_of_type=bool)

        super(
            ClassifierWeightedAverage,
            self).__init__(
            name=self.name,
            settings=self.settings,
            kind=self.kind)