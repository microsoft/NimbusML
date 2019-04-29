# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
# - Generated by tools/entrypoint_compiler.py: do not edit by hand
"""
GamRegressor
"""

__all__ = ["GamRegressor"]


from ...entrypoints.trainers_generalizedadditivemodelregressor import \
    trainers_generalizedadditivemodelregressor
from ...utils.utils import trace
from ..base_pipeline_item import BasePipelineItem, DefaultSignature


class GamRegressor(BasePipelineItem, DefaultSignature):
    """

    Generalized Additive Models

    .. remarks::
        `Generalized additive models
        <https://en.wikipedia.org/wiki/Generalized_additive_model>`_
        (referred
        to throughout as GAM) is a class of models expressable as an
        independent
        sum of individual functions. ``nimbusml``'s GAM learner comes in both
        binary
        classification (using logit-boosting) and regression (using least
        squares) flavors.

        In contrast to many formal definitions of GAM, this implementation
        found
        it convenient to represent learning over stepwise functions, which
        betrays the intention that GAM's components be smooth functions. In
        particular: the learner first discretizes features, and the "step"
        functions learned will step between the discretization boundaries.

        This implementation is based on the this `paper
        <http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.352.7619>`_,
        but diverges from it in several important respects: most
        significantly,
        in each round of boosting, rather than do one feature at a time, it
        instead makes a round on all features simultaneously. In each round,
        it
        will choose only one split point of each feature to change.

        In its current form, the GAM learner has the following advantages and
        disadvantages: on the one hand, they offer ready interpretability
        combined with expressive power, but on the other, they are currently
        slow. We would recommend their usage in the case where the key
        criteria
        is interpretability.

        Let's talk a bit more about interpretabilty. The next most
        interpretable
        model, we might say, is a linear model. But really, let's say that
        you
        have a feature with a coefficient of 3.9293, or something. What do
        you
        know? You know that generally, perhaps, larger values for that
        feature
        are "better." Great. But is 4 better than 3? Is 5 better than 4? To
        what
        degree? Are there "shapes" in the distributions hidden because of the
        reduction of a complex quantity to a single values? These are
        questions
        a linear model fundamentally cannot answer, but a GAM model might.


        **Reference**

            `Generalized additive models
            <https://en.wikipedia.org/wiki/Generalized_additive_model>`_,
            `Intelligible Models for Classification and Regression
            <http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.352.7619>`_


    :param number_of_iterations: Total number of iterations over all features.

    :param feature: Column to use for features.

    :param minimum_example_count_per_leaf: Minimum number of training instances
        required to form a partition.

    :param label: Column to use for labels.

    :param learning_rate: The learning rate.

    :param weight: Column to use for example weight.

    :param normalize: Specifies the type of automatic normalization used:

        * ``"Auto"``: if normalization is needed, it is performed
          automatically. This is the default choice.
        * ``"No"``: no normalization is performed.
        * ``"Yes"``: normalization is performed.
        * ``"Warn"``: if normalization is needed, a warning
          message is displayed, but normalization is not performed.

        Normalization rescales disparate data ranges to a standard scale.
        Feature
        scaling insures the distances between data points are proportional
        and
        enables various optimization methods such as gradient descent to
        converge
        much faster. If normalization is performed, a ``MaxMin`` normalizer
        is
        used. It normalizes values in an interval [a, b] where ``-1 <= a <=
        0``
        and ``0 <= b <= 1`` and ``b - a = 1``. This normalizer preserves
        sparsity by mapping zero to zero.

    :param caching: Whether trainer should cache input training data.

    :param pruning_metrics: Metric for pruning. (For regression, 1: L1, 2:L2;
        default L2).

    :param entropy_coefficient: The entropy (regularization) coefficient
        between 0 and 1.

    :param gain_conf_level: Tree fitting gain confidence requirement (should be
        in the range [0,1) ).

    :param number_of_threads: The number of threads to use.

    :param disk_transpose: Whether to utilize the disk or the data's native
        transposition facilities (where applicable) when performing the
        transpose.

    :param maximum_bin_count_per_feature: Maximum number of distinct values
        (bins) per feature.

    :param maximum_tree_output: Upper bound on absolute value of single output.

    :param get_derivatives_sample_rate: Sample each query 1 in k times in the
        GetDerivatives function.

    :param random_state: The seed of the random number generator.

    :param feature_flocks: Whether to collectivize features during dataset
        preparation to speed up training.

    :param enable_pruning: Enable post-training pruning to avoid overfitting.
        (a validation set is required).

    :param params: Additional arguments sent to compute engine.

    .. seealso::
        :py:func:`FastLinearRegressor
        <nimbusml.linear_model.FastLinearRegressor>`,
        :py:func:`OnlineGradientDescentRegressor
        <nimbusml.linear_model.OnlineGradientDescentRegressor>`


    .. index:: models, classification, svm

    Example:
       .. literalinclude:: /../nimbusml/examples/GamRegressor.py
              :language: python
    """

    @trace
    def __init__(
            self,
            number_of_iterations=9500,
            feature='Features',
            minimum_example_count_per_leaf=10,
            label='Label',
            learning_rate=0.002,
            weight=None,
            normalize='Auto',
            caching='Auto',
            pruning_metrics=2,
            entropy_coefficient=0.0,
            gain_conf_level=0,
            number_of_threads=None,
            disk_transpose=None,
            maximum_bin_count_per_feature=255,
            maximum_tree_output=float('inf'),
            get_derivatives_sample_rate=1,
            random_state=123,
            feature_flocks=True,
            enable_pruning=True,
            **params):
        BasePipelineItem.__init__(
            self, type='regressor', **params)

        self.number_of_iterations = number_of_iterations
        self.feature = feature
        self.minimum_example_count_per_leaf = minimum_example_count_per_leaf
        self.label = label
        self.learning_rate = learning_rate
        self.weight = weight
        self.normalize = normalize
        self.caching = caching
        self.pruning_metrics = pruning_metrics
        self.entropy_coefficient = entropy_coefficient
        self.gain_conf_level = gain_conf_level
        self.number_of_threads = number_of_threads
        self.disk_transpose = disk_transpose
        self.maximum_bin_count_per_feature = maximum_bin_count_per_feature
        self.maximum_tree_output = maximum_tree_output
        self.get_derivatives_sample_rate = get_derivatives_sample_rate
        self.random_state = random_state
        self.feature_flocks = feature_flocks
        self.enable_pruning = enable_pruning

    @property
    def _entrypoint(self):
        return trainers_generalizedadditivemodelregressor

    @trace
    def _get_node(self, **all_args):
        algo_args = dict(
            number_of_iterations=self.number_of_iterations,
            feature_column_name=self.feature,
            minimum_example_count_per_leaf=self.minimum_example_count_per_leaf,
            label_column_name=self.label,
            learning_rate=self.learning_rate,
            example_weight_column_name=self.weight,
            normalize_features=self.normalize,
            caching=self.caching,
            pruning_metrics=self.pruning_metrics,
            entropy_coefficient=self.entropy_coefficient,
            gain_confidence_level=self.gain_conf_level,
            number_of_threads=self.number_of_threads,
            disk_transpose=self.disk_transpose,
            maximum_bin_count_per_feature=self.maximum_bin_count_per_feature,
            maximum_tree_output=self.maximum_tree_output,
            get_derivatives_sample_rate=self.get_derivatives_sample_rate,
            seed=self.random_state,
            feature_flocks=self.feature_flocks,
            enable_pruning=self.enable_pruning)

        all_args.update(algo_args)
        return self._entrypoint(**all_args)
