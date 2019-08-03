# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
# - Generated by tools/entrypoint_compiler.py: do not edit by hand
"""
EnsembleClassifier
"""

__all__ = ["EnsembleClassifier"]


from ...entrypoints._ensemblemulticlassoutputcombiner_multimedian import \
    multi_median
from ...entrypoints._ensemblemulticlasssubmodelselector_allselectormulticlass import \
    all_selector_multi_class
from ...entrypoints._ensemblesubsetselector_bootstrapselector import \
    bootstrap_selector
from ...entrypoints.trainers_ensembleclassification import \
    trainers_ensembleclassification
from ...utils.utils import trace
from ..base_pipeline_item import BasePipelineItem, DefaultSignatureWithRoles


class EnsembleClassifier(
        BasePipelineItem,
        DefaultSignatureWithRoles):
    """

    **Description**
    Train a multi class ensemble model

    .. remarks::



    :param sampling_type: .

    :param num_models: .

    :param sub_model_selector_type: :output_combiner:.

    :param output_combiner: Output combiner.

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

    :param caching: Whether trainer should cache input training data.

    :param train_parallel: All the base learners will run asynchronously if the
        value is true.

    :param batch_size: Batch size.

    :param show_metrics: True, if metrics for each model need to be evaluated
        and shown in comparison table. This is done by using validation set if
        available or the training set.

    :param params: Additional arguments sent to compute engine.

    .. seealso::
        :py:class:`EnsembleRegressor
        <nimbusml.ensemble.EnsembleRegressor>`,
        :py:class:`EnsembleClassifier
        <nimbusml.ensemble.EnsembleClassifier>`


    .. index:: models, linear, SDCA, stochastic, classification, regression

    Example:
       .. literalinclude:: /../nimbusml/examples/FEnsembleClassifier.py
              :language: python
    """

    @trace
    def __init__(
            self,
            sampling_type=bootstrap_selector(
                feature_selector={
                    'Name': 'AllFeatureSelector'}),
        num_models=None,
        sub_model_selector_type=None,
        output_combiner=None,
        normalize='Auto',
        caching='Auto',
        train_parallel=False,
        batch_size=-1,
        show_metrics=False,
            **params):
        BasePipelineItem.__init__(
            self, type='classifier', **params)

        self.sampling_type = sampling_type
        self.num_models = num_models
        self.sub_model_selector_type = sub_model_selector_type
        self.output_combiner = output_combiner
        self.normalize = normalize
        self.caching = caching
        self.train_parallel = train_parallel
        self.batch_size = batch_size
        self.show_metrics = show_metrics

    @property
    def _entrypoint(self):
        return trainers_ensembleclassification

    @trace
    def _get_node(self, **all_args):
        algo_args = dict(
            feature_column_name=self._getattr_role(
                'feature_column_name',
                all_args),
            label_column_name=self._getattr_role(
                'label_column_name',
                all_args),
            sampling_type=self.sampling_type,
            num_models=self.num_models,
            sub_model_selector_type=self.sub_model_selector_type,
            output_combiner=self.output_combiner,
            normalize_features=self.normalize,
            caching=self.caching,
            train_parallel=self.train_parallel,
            batch_size=self.batch_size,
            show_metrics=self.show_metrics)

        all_args.update(algo_args)
        return self._entrypoint(**all_args)
