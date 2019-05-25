# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
# - Generated by tools/entrypoint_compiler.py: do not edit by hand
"""
FastTreesTweedieRegressor
"""

__all__ = ["FastTreesTweedieRegressor"]


from sklearn.base import RegressorMixin

from ..base_predictor import BasePredictor
from ..internal.core.ensemble.fasttreestweedieregressor import \
    FastTreesTweedieRegressor as core
from ..internal.utils.utils import trace


class FastTreesTweedieRegressor(
        core,
        BasePredictor,
        RegressorMixin):
    """

    Machine Learning Fast Tree

    .. remarks::
        Trains gradient boosted decision trees to fit target values using a
        Tweedie loss function. This learner is a generalization of Poisson,
        compound Poisson, and gamma regression.


        **Reference**

            `Wikipedia: Gradient boosting (Gradient tree boosting)
            <https://en.wikipedia.org/wiki/Gradient_boosting#Gradient_tree_boosting>`_

            `Greedy function approximation: A gradient boosting machine.
            <http://projecteuclid.org/DPubS?service=UI&version=1.0&verb=Display&handle=euclid.aos/1013203451>`_

    :param feature: see `Columns </nimbusml/concepts/columns>`_.

    :param group_id: see `Columns </nimbusml/concepts/columns>`_.

    :param label: see `Columns </nimbusml/concepts/columns>`_.

    :param weight: see `Columns </nimbusml/concepts/columns>`_.

    :param number_of_trees: Specifies the total number of decision trees to
        create in the ensemble. By creating more decision trees, you can
        potentially get better coverage, but the training time increases.

    :param number_of_leaves: The maximum number of leaves (terminal nodes) that
        can be created in any tree. Higher values potentially increase the size
        of the tree and get better precision, but risk overfitting and
        requiring longer training times.

    :param min_split: Minimum number of training instances required to form a
        leaf. That is, the minimal number of documents allowed in a leaf of
        regression tree, out of the sub-sampled data. A 'split' means that
        features in each level of the tree (node) are randomly divided.

    :param learning_rate: Determines the size of the step taken in the
        direction of the gradient in each step of the learning process.  This
        determines how fast or slow the learner converges on the optimal
        solution. If the step size is too big, you might overshoot the optimal
        solution.  If the step size is too small, training takes longer to
        converge to the best solution.

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

    :param index: Index parameter for the Tweedie distribution, in the range
        [1, 2]. 1 is Poisson loss, 2 is gamma loss, and intermediate values are
        compound Poisson loss.

    :param best_step_trees: Option for using best regression step trees.

    :param use_line_search: Should we use line search for a step size.

    :param maximum_number_of_line_search_steps: Number of post-bracket line
        search steps.

    :param minimum_step_size: Minimum line search step size.

    :param optimizer: Default is ``sgd``.

    :param early_stopping_rule: Early stopping rule. (Validation set (/valid)
        is required.).

    :param early_stopping_metrics: Early stopping metrics. (For regression, 1:
        L1, 2:L2; for ranking, 1:NDCG@1, 3:NDCG@3).

    :param enable_pruning: Enable post-training pruning to avoid overfitting.
        (a validation set is required).

    :param use_tolerant_pruning: Use window and tolerance for pruning.

    :param pruning_threshold: The tolerance threshold for pruning.

    :param pruning_window_size: The moving window size for pruning.

    :param shrinkage: Shrinkage.

    :param dropout_rate: Dropout rate for tree regularization.

    :param get_derivatives_sample_rate: Sample each query 1 in k times in the
        GetDerivatives function.

    :param write_last_ensemble: Write the last ensemble instead of the one
        determined by early stopping.

    :param maximum_tree_output: Upper bound on absolute value of single tree
        output.

    :param random_start: Training starts from random ordering (determined by
        /r1).

    :param filter_zero_lambdas: Filter zero lambdas during training.

    :param baseline_scores_formula: Freeform defining the scores that should be
        used as the baseline ranker.

    :param baseline_alpha_risk: Baseline alpha for tradeoffs of risk (0 is
        normal training).

    :param position_discount_freeform: The discount freeform which specifies
        the per position discounts of examples in a query (uses a single
        variable P for position where P=0 is first position).

    :param parallel_trainer: Allows to choose Parallel FastTree Learning
        Algorithm.

    :param number_of_threads: The number of threads to use.

    :param random_state: The seed of the random number generator.

    :param feature_selection_seed: The seed of the active feature selection.

    :param entropy_coefficient: The entropy (regularization) coefficient
        between 0 and 1.

    :param histogram_pool_size: The number of histograms in the pool (between 2
        and numLeaves).

    :param disk_transpose: Whether to utilize the disk or the data's native
        transposition facilities (where applicable) when performing the
        transpose.

    :param feature_flocks: Whether to collectivize features during dataset
        preparation to speed up training.

    :param categorical_split: Whether to do split based on multiple categorical
        feature values.

    :param maximum_categorical_group_count_per_node: Maximum categorical split
        groups to consider when splitting on a categorical feature. Split
        groups are a collection of split points. This is used to reduce
        overfitting when there many categorical features.

    :param maximum_categorical_split_point_count: Maximum categorical split
        points to consider when splitting on a categorical feature.

    :param minimum_example_fraction_for_categorical_split: Minimum categorical
        example percentage in a bin to consider for a split.

    :param minimum_examples_for_categorical_split: Minimum categorical example
        count in a bin to consider for a split.

    :param bias: Bias for calculating gradient for each feature bin for a
        categorical feature.

    :param bundling: Bundle low population bins. Bundle.None(0): no bundling,
        Bundle.AggregateLowPopulation(1): Bundle low population,
        Bundle.Adjacent(2): Neighbor low population bundle.

    :param maximum_bin_count_per_feature: Maximum number of distinct values
        (bins) per feature.

    :param sparsify_threshold: Sparsity level needed to use sparse feature
        representation.

    :param first_use_penalty: The feature first use penalty coefficient. This
        is a form of regularization that incurs a penalty for using a new
        feature when creating the tree. Increase this value to create trees
        that don't use many features.

    :param feature_reuse_penalty: The feature re-use penalty (regularization)
        coefficient.

    :param gain_conf_level: Tree fitting gain confidence requirement (should be
        in the range [0,1) ).

    :param softmax_temperature: The temperature of the randomized softmax
        distribution for choosing the feature.

    :param execution_time: Print execution time breakdown to stdout.

    :param feature_fraction: The fraction of features (chosen randomly) to use
        on each iteration.

    :param bagging_size: Number of trees in each bag (0 for disabling bagging).

    :param bagging_example_fraction: Percentage of training examples used in
        each bag.

    :param feature_fraction_per_split: The fraction of features (chosen
        randomly) to use on each split.

    :param smoothing: Smoothing paramter for tree regularization.

    :param allow_empty_trees: When a root split is impossible, allow training
        to proceed.

    :param feature_compression_level: The level of feature compression to use.

    :param compress_ensemble: Compress the tree Ensemble.

    :param test_frequency: Calculate metric values for train/valid/test every k
        rounds.

    :param params: Additional arguments sent to compute engine.

    .. seealso::
        :py:func:`FastForestRegressor
        <nimbusml.ensemble.FastForestRegressor>`,
        :py:func:`FastTreesBinaryClassifier
        <nimbusml.ensemble.FastTreesBinaryClassifier>`,

    .. index:: models, regression

    Example:
       .. literalinclude:: /../nimbusml/examples/FastTreesTweedieRegressor.py
              :language: python

    """

    @trace
    def __init__(
            self,
            number_of_trees=100,
            number_of_leaves=20,
            min_split=10,
            learning_rate=0.2,
            normalize='Auto',
            caching='Auto',
            index=1.5,
            best_step_trees=False,
            use_line_search=False,
            maximum_number_of_line_search_steps=0,
            minimum_step_size=0.0,
            optimizer='GradientDescent',
            early_stopping_rule=None,
            early_stopping_metrics=1,
            enable_pruning=False,
            use_tolerant_pruning=False,
            pruning_threshold=0.004,
            pruning_window_size=5,
            shrinkage=1.0,
            dropout_rate=0.0,
            get_derivatives_sample_rate=1,
            write_last_ensemble=False,
            maximum_tree_output=100.0,
            random_start=False,
            filter_zero_lambdas=False,
            baseline_scores_formula=None,
            baseline_alpha_risk=None,
            position_discount_freeform=None,
            parallel_trainer=None,
            number_of_threads=None,
            random_state=123,
            feature_selection_seed=123,
            entropy_coefficient=0.0,
            histogram_pool_size=-1,
            disk_transpose=None,
            feature_flocks=True,
            categorical_split=False,
            maximum_categorical_group_count_per_node=64,
            maximum_categorical_split_point_count=64,
            minimum_example_fraction_for_categorical_split=0.001,
            minimum_examples_for_categorical_split=100,
            bias=0.0,
            bundling='None',
            maximum_bin_count_per_feature=255,
            sparsify_threshold=0.7,
            first_use_penalty=0.0,
            feature_reuse_penalty=0.0,
            gain_conf_level=0.0,
            softmax_temperature=0.0,
            execution_time=False,
            feature_fraction=1.0,
            bagging_size=0,
            bagging_example_fraction=0.7,
            feature_fraction_per_split=1.0,
            smoothing=0.0,
            allow_empty_trees=True,
            feature_compression_level=1,
            compress_ensemble=False,
            test_frequency=2147483647,
            feature=None,
            group_id=None,
            label=None,
            weight=None,
            **params):

        if 'feature_column_name' in params:
            raise NameError(
                "'feature_column_name' must be renamed to 'feature'")
        if feature:
            params['feature_column_name'] = feature
        if 'row_group_column_name' in params:
            raise NameError(
                "'row_group_column_name' must be renamed to 'group_id'")
        if group_id:
            params['row_group_column_name'] = group_id
        if 'label_column_name' in params:
            raise NameError(
                "'label_column_name' must be renamed to 'label'")
        if label:
            params['label_column_name'] = label
        if 'example_weight_column_name' in params:
            raise NameError(
                "'example_weight_column_name' must be renamed to 'weight'")
        if weight:
            params['example_weight_column_name'] = weight
        BasePredictor.__init__(self, type='regressor', **params)
        core.__init__(
            self,
            number_of_trees=number_of_trees,
            number_of_leaves=number_of_leaves,
            min_split=min_split,
            learning_rate=learning_rate,
            normalize=normalize,
            caching=caching,
            index=index,
            best_step_trees=best_step_trees,
            use_line_search=use_line_search,
            maximum_number_of_line_search_steps=maximum_number_of_line_search_steps,
            minimum_step_size=minimum_step_size,
            optimizer=optimizer,
            early_stopping_rule=early_stopping_rule,
            early_stopping_metrics=early_stopping_metrics,
            enable_pruning=enable_pruning,
            use_tolerant_pruning=use_tolerant_pruning,
            pruning_threshold=pruning_threshold,
            pruning_window_size=pruning_window_size,
            shrinkage=shrinkage,
            dropout_rate=dropout_rate,
            get_derivatives_sample_rate=get_derivatives_sample_rate,
            write_last_ensemble=write_last_ensemble,
            maximum_tree_output=maximum_tree_output,
            random_start=random_start,
            filter_zero_lambdas=filter_zero_lambdas,
            baseline_scores_formula=baseline_scores_formula,
            baseline_alpha_risk=baseline_alpha_risk,
            position_discount_freeform=position_discount_freeform,
            parallel_trainer=parallel_trainer,
            number_of_threads=number_of_threads,
            random_state=random_state,
            feature_selection_seed=feature_selection_seed,
            entropy_coefficient=entropy_coefficient,
            histogram_pool_size=histogram_pool_size,
            disk_transpose=disk_transpose,
            feature_flocks=feature_flocks,
            categorical_split=categorical_split,
            maximum_categorical_group_count_per_node=maximum_categorical_group_count_per_node,
            maximum_categorical_split_point_count=maximum_categorical_split_point_count,
            minimum_example_fraction_for_categorical_split=minimum_example_fraction_for_categorical_split,
            minimum_examples_for_categorical_split=minimum_examples_for_categorical_split,
            bias=bias,
            bundling=bundling,
            maximum_bin_count_per_feature=maximum_bin_count_per_feature,
            sparsify_threshold=sparsify_threshold,
            first_use_penalty=first_use_penalty,
            feature_reuse_penalty=feature_reuse_penalty,
            gain_conf_level=gain_conf_level,
            softmax_temperature=softmax_temperature,
            execution_time=execution_time,
            feature_fraction=feature_fraction,
            bagging_size=bagging_size,
            bagging_example_fraction=bagging_example_fraction,
            feature_fraction_per_split=feature_fraction_per_split,
            smoothing=smoothing,
            allow_empty_trees=allow_empty_trees,
            feature_compression_level=feature_compression_level,
            compress_ensemble=compress_ensemble,
            test_frequency=test_frequency,
            **params)
        self.feature = feature
        self.group_id = group_id
        self.label = label
        self.weight = weight

    def get_params(self, deep=False):
        """
        Get the parameters for this operator.
        """
        return core.get_params(self)
