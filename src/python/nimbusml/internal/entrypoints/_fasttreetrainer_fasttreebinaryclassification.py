# - Generated by tools/entrypoint_compiler.py: do not edit by hand
"""
FastTreeBinaryClassification
"""

import numbers

from ..utils.entrypoints import Component
from ..utils.utils import try_set


def fast_tree_binary_classification(
        training_data,
        number_of_trees=100,
        number_of_leaves=20,
        feature_column_name='Features',
        minimum_example_count_per_leaf=10,
        label_column_name='Label',
        learning_rate=0.2,
        example_weight_column_name=None,
        row_group_column_name=None,
        normalize_features='Auto',
        caching='Auto',
        unbalanced_sets=False,
        best_step_ranking_regression_trees=False,
        use_line_search=False,
        maximum_number_of_line_search_steps=0,
        minimum_step_size=0.0,
        optimization_algorithm='GradientDescent',
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
        seed=123,
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
        feature_first_use_penalty=0.0,
        feature_reuse_penalty=0.0,
        gain_confidence_level=0.0,
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
        print_test_graph=False,
        print_train_valid_graph=False,
        test_frequency=2147483647,
        **params):
    """
    **Description**
        Uses a logit-boost boosted tree learner to perform binary
        classification.

    :param number_of_trees: Total number of decision trees to create
        in the ensemble (settings).
    :param training_data: The data to be used for training
        (settings).
    :param number_of_leaves: The max number of leaves in each
        regression tree (settings).
    :param feature_column_name: Column to use for features
        (settings).
    :param minimum_example_count_per_leaf: The minimal number of
        examples allowed in a leaf of a regression tree, out of the
        subsampled data (settings).
    :param label_column_name: Column to use for labels (settings).
    :param learning_rate: The learning rate (settings).
    :param example_weight_column_name: Column to use for example
        weight (settings).
    :param row_group_column_name: Column to use for example groupId
        (settings).
    :param normalize_features: Normalize option for the feature
        column (settings).
    :param caching: Whether trainer should cache input training data
        (settings).
    :param unbalanced_sets: Option for using derivatives optimized
        for unbalanced sets (settings).
    :param best_step_ranking_regression_trees: Option for using best
        regression step trees (settings).
    :param use_line_search: Should we use line search for a step size
        (settings).
    :param maximum_number_of_line_search_steps: Number of post-
        bracket line search steps (settings).
    :param minimum_step_size: Minimum line search step size
        (settings).
    :param optimization_algorithm: Optimization algorithm to be used
        (GradientDescent, AcceleratedGradientDescent) (settings).
    :param early_stopping_rule: Early stopping rule. (Validation set
        (/valid) is required.) (settings).
    :param early_stopping_metrics: Early stopping metrics. (For
        regression, 1: L1, 2:L2; for ranking, 1:NDCG@1, 3:NDCG@3)
        (settings).
    :param enable_pruning: Enable post-training pruning to avoid
        overfitting. (a validation set is required) (settings).
    :param use_tolerant_pruning: Use window and tolerance for pruning
        (settings).
    :param pruning_threshold: The tolerance threshold for pruning
        (settings).
    :param pruning_window_size: The moving window size for pruning
        (settings).
    :param shrinkage: Shrinkage (settings).
    :param dropout_rate: Dropout rate for tree regularization
        (settings).
    :param get_derivatives_sample_rate: Sample each query 1 in k
        times in the GetDerivatives function (settings).
    :param write_last_ensemble: Write the last ensemble instead of
        the one determined by early stopping (settings).
    :param maximum_tree_output: Upper bound on absolute value of
        single tree output (settings).
    :param random_start: Training starts from random ordering
        (determined by /r1) (settings).
    :param filter_zero_lambdas: Filter zero lambdas during training
        (settings).
    :param baseline_scores_formula: Freeform defining the scores that
        should be used as the baseline ranker (settings).
    :param baseline_alpha_risk: Baseline alpha for tradeoffs of risk
        (0 is normal training) (settings).
    :param position_discount_freeform: The discount freeform which
        specifies the per position discounts of examples in a query
        (uses a single variable P for position where P=0 is first
        position) (settings).
    :param parallel_trainer: Allows to choose Parallel FastTree
        Learning Algorithm (settings).
    :param number_of_threads: The number of threads to use
        (settings).
    :param seed: The seed of the random number generator (settings).
    :param feature_selection_seed: The seed of the active feature
        selection (settings).
    :param entropy_coefficient: The entropy (regularization)
        coefficient between 0 and 1 (settings).
    :param histogram_pool_size: The number of histograms in the pool
        (between 2 and numLeaves) (settings).
    :param disk_transpose: Whether to utilize the disk or the data's
        native transposition facilities (where applicable) when
        performing the transpose (settings).
    :param feature_flocks: Whether to collectivize features during
        dataset preparation to speed up training (settings).
    :param categorical_split: Whether to do split based on multiple
        categorical feature values. (settings).
    :param maximum_categorical_group_count_per_node: Maximum
        categorical split groups to consider when splitting on a
        categorical feature. Split groups are a collection of split
        points. This is used to reduce overfitting when there many
        categorical features. (settings).
    :param maximum_categorical_split_point_count: Maximum categorical
        split points to consider when splitting on a categorical
        feature. (settings).
    :param minimum_example_fraction_for_categorical_split: Minimum
        categorical example percentage in a bin to consider for a
        split. (settings).
    :param minimum_examples_for_categorical_split: Minimum
        categorical example count in a bin to consider for a split.
        (settings).
    :param bias: Bias for calculating gradient for each feature bin
        for a categorical feature. (settings).
    :param bundling: Bundle low population bins. Bundle.None(0): no
        bundling, Bundle.AggregateLowPopulation(1): Bundle low
        population, Bundle.Adjacent(2): Neighbor low population
        bundle. (settings).
    :param maximum_bin_count_per_feature: Maximum number of distinct
        values (bins) per feature (settings).
    :param sparsify_threshold: Sparsity level needed to use sparse
        feature representation (settings).
    :param feature_first_use_penalty: The feature first use penalty
        coefficient (settings).
    :param feature_reuse_penalty: The feature re-use penalty
        (regularization) coefficient (settings).
    :param gain_confidence_level: Tree fitting gain confidence
        requirement (should be in the range [0,1) ). (settings).
    :param softmax_temperature: The temperature of the randomized
        softmax distribution for choosing the feature (settings).
    :param execution_time: Print execution time breakdown to stdout
        (settings).
    :param feature_fraction: The fraction of features (chosen
        randomly) to use on each iteration (settings).
    :param bagging_size: Number of trees in each bag (0 for disabling
        bagging) (settings).
    :param bagging_example_fraction: Percentage of training examples
        used in each bag (settings).
    :param feature_fraction_per_split: The fraction of features
        (chosen randomly) to use on each split (settings).
    :param smoothing: Smoothing paramter for tree regularization
        (settings).
    :param allow_empty_trees: When a root split is impossible, allow
        training to proceed (settings).
    :param feature_compression_level: The level of feature
        compression to use (settings).
    :param compress_ensemble: Compress the tree Ensemble (settings).
    :param print_test_graph: Print metrics graph for the first test
        set (settings).
    :param print_train_valid_graph: Print Train and Validation
        metrics in graph (settings).
    :param test_frequency: Calculate metric values for
        train/valid/test every k rounds (settings).
    """

    entrypoint_name = 'FastTreeBinaryClassification'
    settings = {}

    if number_of_trees is not None:
        settings['NumberOfTrees'] = try_set(
            obj=number_of_trees,
            none_acceptable=True,
            is_of_type=numbers.Real)
    if training_data is not None:
        settings['TrainingData'] = try_set(
            obj=training_data, none_acceptable=False, is_of_type=str)
    if number_of_leaves is not None:
        settings['NumberOfLeaves'] = try_set(
            obj=number_of_leaves,
            none_acceptable=True,
            is_of_type=numbers.Real)
    if feature_column_name is not None:
        settings['FeatureColumnName'] = try_set(
            obj=feature_column_name,
            none_acceptable=True,
            is_of_type=str,
            is_column=True)
    if minimum_example_count_per_leaf is not None:
        settings['MinimumExampleCountPerLeaf'] = try_set(
            obj=minimum_example_count_per_leaf,
            none_acceptable=True,
            is_of_type=numbers.Real)
    if label_column_name is not None:
        settings['LabelColumnName'] = try_set(
            obj=label_column_name,
            none_acceptable=True,
            is_of_type=str,
            is_column=True)
    if learning_rate is not None:
        settings['LearningRate'] = try_set(
            obj=learning_rate,
            none_acceptable=True,
            is_of_type=numbers.Real)
    if example_weight_column_name is not None:
        settings['ExampleWeightColumnName'] = try_set(
            obj=example_weight_column_name,
            none_acceptable=True,
            is_of_type=str,
            is_column=True)
    if row_group_column_name is not None:
        settings['RowGroupColumnName'] = try_set(
            obj=row_group_column_name,
            none_acceptable=True,
            is_of_type=str,
            is_column=True)
    if normalize_features is not None:
        settings['NormalizeFeatures'] = try_set(
            obj=normalize_features,
            none_acceptable=True,
            is_of_type=str,
            values=[
                'No',
                'Warn',
                'Auto',
                'Yes'])
    if caching is not None:
        settings['Caching'] = try_set(
            obj=caching,
            none_acceptable=True,
            is_of_type=str,
            values=[
                'Auto',
                'Memory',
                'None'])
    if unbalanced_sets is not None:
        settings['UnbalancedSets'] = try_set(
            obj=unbalanced_sets, none_acceptable=True, is_of_type=bool)
    if best_step_ranking_regression_trees is not None:
        settings['BestStepRankingRegressionTrees'] = try_set(
            obj=best_step_ranking_regression_trees,
            none_acceptable=True,
            is_of_type=bool)
    if use_line_search is not None:
        settings['UseLineSearch'] = try_set(
            obj=use_line_search, none_acceptable=True, is_of_type=bool)
    if maximum_number_of_line_search_steps is not None:
        settings['MaximumNumberOfLineSearchSteps'] = try_set(
            obj=maximum_number_of_line_search_steps,
            none_acceptable=True,
            is_of_type=numbers.Real)
    if minimum_step_size is not None:
        settings['MinimumStepSize'] = try_set(
            obj=minimum_step_size,
            none_acceptable=True,
            is_of_type=numbers.Real)
    if optimization_algorithm is not None:
        settings['OptimizationAlgorithm'] = try_set(
            obj=optimization_algorithm,
            none_acceptable=True,
            is_of_type=str,
            values=[
                'GradientDescent',
                'AcceleratedGradientDescent',
                'ConjugateGradientDescent'])
    if early_stopping_rule is not None:
        settings['EarlyStoppingRule'] = try_set(
            obj=early_stopping_rule, none_acceptable=True, is_of_type=dict)
    if early_stopping_metrics is not None:
        settings['EarlyStoppingMetrics'] = try_set(
            obj=early_stopping_metrics,
            none_acceptable=True,
            is_of_type=numbers.Real)
    if enable_pruning is not None:
        settings['EnablePruning'] = try_set(
            obj=enable_pruning, none_acceptable=True, is_of_type=bool)
    if use_tolerant_pruning is not None:
        settings['UseTolerantPruning'] = try_set(
            obj=use_tolerant_pruning, none_acceptable=True, is_of_type=bool)
    if pruning_threshold is not None:
        settings['PruningThreshold'] = try_set(
            obj=pruning_threshold,
            none_acceptable=True,
            is_of_type=numbers.Real)
    if pruning_window_size is not None:
        settings['PruningWindowSize'] = try_set(
            obj=pruning_window_size,
            none_acceptable=True,
            is_of_type=numbers.Real)
    if shrinkage is not None:
        settings['Shrinkage'] = try_set(
            obj=shrinkage,
            none_acceptable=True,
            is_of_type=numbers.Real)
    if dropout_rate is not None:
        settings['DropoutRate'] = try_set(
            obj=dropout_rate,
            none_acceptable=True,
            is_of_type=numbers.Real)
    if get_derivatives_sample_rate is not None:
        settings['GetDerivativesSampleRate'] = try_set(
            obj=get_derivatives_sample_rate,
            none_acceptable=True,
            is_of_type=numbers.Real)
    if write_last_ensemble is not None:
        settings['WriteLastEnsemble'] = try_set(
            obj=write_last_ensemble, none_acceptable=True, is_of_type=bool)
    if maximum_tree_output is not None:
        settings['MaximumTreeOutput'] = try_set(
            obj=maximum_tree_output,
            none_acceptable=True,
            is_of_type=numbers.Real)
    if random_start is not None:
        settings['RandomStart'] = try_set(
            obj=random_start, none_acceptable=True, is_of_type=bool)
    if filter_zero_lambdas is not None:
        settings['FilterZeroLambdas'] = try_set(
            obj=filter_zero_lambdas, none_acceptable=True, is_of_type=bool)
    if baseline_scores_formula is not None:
        settings['BaselineScoresFormula'] = try_set(
            obj=baseline_scores_formula, none_acceptable=True, is_of_type=str)
    if baseline_alpha_risk is not None:
        settings['BaselineAlphaRisk'] = try_set(
            obj=baseline_alpha_risk, none_acceptable=True, is_of_type=str)
    if position_discount_freeform is not None:
        settings['PositionDiscountFreeform'] = try_set(
            obj=position_discount_freeform,
            none_acceptable=True,
            is_of_type=str)
    if parallel_trainer is not None:
        settings['ParallelTrainer'] = try_set(
            obj=parallel_trainer, none_acceptable=True, is_of_type=dict)
    if number_of_threads is not None:
        settings['NumberOfThreads'] = try_set(
            obj=number_of_threads,
            none_acceptable=True,
            is_of_type=numbers.Real)
    if seed is not None:
        settings['Seed'] = try_set(
            obj=seed,
            none_acceptable=True,
            is_of_type=numbers.Real)
    if feature_selection_seed is not None:
        settings['FeatureSelectionSeed'] = try_set(
            obj=feature_selection_seed,
            none_acceptable=True,
            is_of_type=numbers.Real)
    if entropy_coefficient is not None:
        settings['EntropyCoefficient'] = try_set(
            obj=entropy_coefficient,
            none_acceptable=True,
            is_of_type=numbers.Real)
    if histogram_pool_size is not None:
        settings['HistogramPoolSize'] = try_set(
            obj=histogram_pool_size,
            none_acceptable=True,
            is_of_type=numbers.Real)
    if disk_transpose is not None:
        settings['DiskTranspose'] = try_set(
            obj=disk_transpose, none_acceptable=True, is_of_type=bool)
    if feature_flocks is not None:
        settings['FeatureFlocks'] = try_set(
            obj=feature_flocks, none_acceptable=True, is_of_type=bool)
    if categorical_split is not None:
        settings['CategoricalSplit'] = try_set(
            obj=categorical_split, none_acceptable=True, is_of_type=bool)
    if maximum_categorical_group_count_per_node is not None:
        settings['MaximumCategoricalGroupCountPerNode'] = try_set(
            obj=maximum_categorical_group_count_per_node,
            none_acceptable=True,
            is_of_type=numbers.Real)
    if maximum_categorical_split_point_count is not None:
        settings['MaximumCategoricalSplitPointCount'] = try_set(
            obj=maximum_categorical_split_point_count,
            none_acceptable=True,
            is_of_type=numbers.Real)
    if minimum_example_fraction_for_categorical_split is not None:
        settings['MinimumExampleFractionForCategoricalSplit'] = try_set(
            obj=minimum_example_fraction_for_categorical_split,
            none_acceptable=True,
            is_of_type=numbers.Real)
    if minimum_examples_for_categorical_split is not None:
        settings['MinimumExamplesForCategoricalSplit'] = try_set(
            obj=minimum_examples_for_categorical_split,
            none_acceptable=True,
            is_of_type=numbers.Real)
    if bias is not None:
        settings['Bias'] = try_set(
            obj=bias,
            none_acceptable=True,
            is_of_type=numbers.Real)
    if bundling is not None:
        settings['Bundling'] = try_set(
            obj=bundling, none_acceptable=True, is_of_type=str, values=[
                'None', 'AggregateLowPopulation', 'Adjacent'])
    if maximum_bin_count_per_feature is not None:
        settings['MaximumBinCountPerFeature'] = try_set(
            obj=maximum_bin_count_per_feature,
            none_acceptable=True,
            is_of_type=numbers.Real)
    if sparsify_threshold is not None:
        settings['SparsifyThreshold'] = try_set(
            obj=sparsify_threshold,
            none_acceptable=True,
            is_of_type=numbers.Real)
    if feature_first_use_penalty is not None:
        settings['FeatureFirstUsePenalty'] = try_set(
            obj=feature_first_use_penalty,
            none_acceptable=True,
            is_of_type=numbers.Real)
    if feature_reuse_penalty is not None:
        settings['FeatureReusePenalty'] = try_set(
            obj=feature_reuse_penalty,
            none_acceptable=True,
            is_of_type=numbers.Real)
    if gain_confidence_level is not None:
        settings['GainConfidenceLevel'] = try_set(
            obj=gain_confidence_level,
            none_acceptable=True,
            is_of_type=numbers.Real)
    if softmax_temperature is not None:
        settings['SoftmaxTemperature'] = try_set(
            obj=softmax_temperature,
            none_acceptable=True,
            is_of_type=numbers.Real)
    if execution_time is not None:
        settings['ExecutionTime'] = try_set(
            obj=execution_time, none_acceptable=True, is_of_type=bool)
    if feature_fraction is not None:
        settings['FeatureFraction'] = try_set(
            obj=feature_fraction,
            none_acceptable=True,
            is_of_type=numbers.Real)
    if bagging_size is not None:
        settings['BaggingSize'] = try_set(
            obj=bagging_size,
            none_acceptable=True,
            is_of_type=numbers.Real)
    if bagging_example_fraction is not None:
        settings['BaggingExampleFraction'] = try_set(
            obj=bagging_example_fraction,
            none_acceptable=True,
            is_of_type=numbers.Real)
    if feature_fraction_per_split is not None:
        settings['FeatureFractionPerSplit'] = try_set(
            obj=feature_fraction_per_split,
            none_acceptable=True,
            is_of_type=numbers.Real)
    if smoothing is not None:
        settings['Smoothing'] = try_set(
            obj=smoothing,
            none_acceptable=True,
            is_of_type=numbers.Real)
    if allow_empty_trees is not None:
        settings['AllowEmptyTrees'] = try_set(
            obj=allow_empty_trees, none_acceptable=True, is_of_type=bool)
    if feature_compression_level is not None:
        settings['FeatureCompressionLevel'] = try_set(
            obj=feature_compression_level,
            none_acceptable=True,
            is_of_type=numbers.Real)
    if compress_ensemble is not None:
        settings['CompressEnsemble'] = try_set(
            obj=compress_ensemble, none_acceptable=True, is_of_type=bool)
    if print_test_graph is not None:
        settings['PrintTestGraph'] = try_set(
            obj=print_test_graph, none_acceptable=True, is_of_type=bool)
    if print_train_valid_graph is not None:
        settings['PrintTrainValidGraph'] = try_set(
            obj=print_train_valid_graph, none_acceptable=True, is_of_type=bool)
    if test_frequency is not None:
        settings['TestFrequency'] = try_set(
            obj=test_frequency,
            none_acceptable=True,
            is_of_type=numbers.Real)

    component = Component(
        name=entrypoint_name,
        settings=settings,
        kind='FastTreeTrainer')
    return component
