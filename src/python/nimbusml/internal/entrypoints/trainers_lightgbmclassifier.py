# - Generated by tools/entrypoint_compiler.py: do not edit by hand
"""
Trainers.LightGbmClassifier
"""

import numbers

from ..utils.entrypoints import EntryPoint
from ..utils.utils import try_set, unlist


def trainers_lightgbmclassifier(
        training_data,
        predictor_model=None,
        num_boost_round=100,
        learning_rate=None,
        num_leaves=None,
        min_data_per_leaf=None,
        feature_column='Features',
        booster=None,
        label_column='Label',
        weight_column=None,
        group_id_column=None,
        normalize_features='Auto',
        caching='Auto',
        max_bin=255,
        verbose_eval=False,
        silent=True,
        n_thread=None,
        eval_metric='DefaultMetric',
        use_softmax=None,
        early_stopping_round=0,
        custom_gains='0,3,7,15,31,63,127,255,511,1023,2047,4095',
        sigmoid=0.5,
        batch_size=1048576,
        use_cat=None,
        use_missing=False,
        min_data_per_group=100,
        max_cat_threshold=32,
        cat_smooth=10.0,
        cat_l2=10.0,
        seed=None,
        parallel_trainer=None,
        **params):
    """
    **Description**
        Train a LightGBM multi class model.

    :param num_boost_round: Number of iterations. (inputs).
    :param training_data: The data to be used for training (inputs).
    :param learning_rate: Shrinkage rate for trees, used to prevent
        over-fitting. Range: (0,1]. (inputs).
    :param num_leaves: Maximum leaves for trees. (inputs).
    :param min_data_per_leaf: Minimum number of instances needed in a
        child. (inputs).
    :param feature_column: Column to use for features (inputs).
    :param booster: Which booster to use, can be gbtree, gblinear or
        dart. gbtree and dart use tree based model while gblinear
        uses linear function. (inputs).
    :param label_column: Column to use for labels (inputs).
    :param weight_column: Column to use for example weight (inputs).
    :param group_id_column: Column to use for example groupId
        (inputs).
    :param normalize_features: Normalize option for the feature
        column (inputs).
    :param caching: Whether learner should cache input training data
        (inputs).
    :param max_bin: Max number of bucket bin for features. (inputs).
    :param verbose_eval: Verbose (inputs).
    :param silent: Printing running messages. (inputs).
    :param n_thread: Number of parallel threads used to run LightGBM.
        (inputs).
    :param eval_metric: Evaluation metrics. (inputs).
    :param use_softmax: Use softmax loss for the multi
        classification. (inputs).
    :param early_stopping_round: Rounds of early stopping, 0 will
        disable it. (inputs).
    :param custom_gains: Comma seperated list of gains associated to
        each relevance label. (inputs).
    :param sigmoid: Parameter for the sigmoid function. Used only in
        LightGbmBinaryTrainer, LightGbmMulticlassTrainer and in
        LightGbmRankingTrainer. (inputs).
    :param batch_size: Number of entries in a batch when loading
        data. (inputs).
    :param use_cat: Enable categorical split or not. (inputs).
    :param use_missing: Enable missing value auto infer or not.
        (inputs).
    :param min_data_per_group: Min number of instances per
        categorical group. (inputs).
    :param max_cat_threshold: Max number of categorical thresholds.
        (inputs).
    :param cat_smooth: Lapalace smooth term in categorical feature
        spilt. Avoid the bias of small categories. (inputs).
    :param cat_l2: L2 Regularization for categorical split. (inputs).
    :param seed: Sets the random seed for LightGBM to use. (inputs).
    :param parallel_trainer: Parallel LightGBM Learning Algorithm
        (inputs).
    :param predictor_model: The trained model (outputs).
    """

    entrypoint_name = 'Trainers.LightGbmClassifier'
    inputs = {}
    outputs = {}

    if num_boost_round is not None:
        inputs['NumBoostRound'] = try_set(
            obj=num_boost_round,
            none_acceptable=True,
            is_of_type=numbers.Real)
    if training_data is not None:
        inputs['TrainingData'] = try_set(
            obj=training_data,
            none_acceptable=False,
            is_of_type=str)
    if learning_rate is not None:
        inputs['LearningRate'] = try_set(
            obj=learning_rate,
            none_acceptable=True,
            is_of_type=numbers.Real)
    if num_leaves is not None:
        inputs['NumLeaves'] = try_set(
            obj=num_leaves,
            none_acceptable=True,
            is_of_type=numbers.Real)
    if min_data_per_leaf is not None:
        inputs['MinDataPerLeaf'] = try_set(
            obj=min_data_per_leaf,
            none_acceptable=True,
            is_of_type=numbers.Real)
    if feature_column is not None:
        inputs['FeatureColumn'] = try_set(
            obj=feature_column,
            none_acceptable=True,
            is_of_type=str,
            is_column=True)
    if booster is not None:
        inputs['Booster'] = try_set(
            obj=booster,
            none_acceptable=True,
            is_of_type=dict)
    if label_column is not None:
        inputs['LabelColumn'] = try_set(
            obj=label_column,
            none_acceptable=True,
            is_of_type=str,
            is_column=True)
    if weight_column is not None:
        inputs['WeightColumn'] = try_set(
            obj=weight_column,
            none_acceptable=True,
            is_of_type=str,
            is_column=True)
    if group_id_column is not None:
        inputs['GroupIdColumn'] = try_set(
            obj=group_id_column,
            none_acceptable=True,
            is_of_type=str,
            is_column=True)
    if normalize_features is not None:
        inputs['NormalizeFeatures'] = try_set(
            obj=normalize_features,
            none_acceptable=True,
            is_of_type=str,
            values=[
                'No',
                'Warn',
                'Auto',
                'Yes'])
    if caching is not None:
        inputs['Caching'] = try_set(
            obj=caching,
            none_acceptable=True,
            is_of_type=str,
            values=[
                'Auto',
                'Memory',
                'None'])
    if max_bin is not None:
        inputs['MaxBin'] = try_set(
            obj=max_bin,
            none_acceptable=True,
            is_of_type=numbers.Real)
    if verbose_eval is not None:
        inputs['VerboseEval'] = try_set(
            obj=verbose_eval,
            none_acceptable=True,
            is_of_type=bool)
    if silent is not None:
        inputs['Silent'] = try_set(
            obj=silent,
            none_acceptable=True,
            is_of_type=bool)
    if n_thread is not None:
        inputs['NThread'] = try_set(
            obj=n_thread,
            none_acceptable=True,
            is_of_type=numbers.Real)
    if eval_metric is not None:
        inputs['EvalMetric'] = try_set(
            obj=eval_metric,
            none_acceptable=True,
            is_of_type=str,
            values=[
                'DefaultMetric',
                'Rmse',
                'Mae',
                'Logloss',
                'Error',
                'Merror',
                'Mlogloss',
                'Auc',
                'Ndcg',
                'Map'])
    if use_softmax is not None:
        inputs['UseSoftmax'] = try_set(
            obj=use_softmax,
            none_acceptable=True,
            is_of_type=bool)
    if early_stopping_round is not None:
        inputs['EarlyStoppingRound'] = try_set(
            obj=early_stopping_round,
            none_acceptable=True,
            is_of_type=numbers.Real)
    if custom_gains is not None:
        inputs['CustomGains'] = try_set(
            obj=custom_gains,
            none_acceptable=True,
            is_of_type=str)
    if sigmoid is not None:
        inputs['Sigmoid'] = try_set(
            obj=sigmoid,
            none_acceptable=True,
            is_of_type=numbers.Real)
    if batch_size is not None:
        inputs['BatchSize'] = try_set(
            obj=batch_size,
            none_acceptable=True,
            is_of_type=numbers.Real)
    if use_cat is not None:
        inputs['UseCat'] = try_set(
            obj=use_cat,
            none_acceptable=True,
            is_of_type=bool)
    if use_missing is not None:
        inputs['UseMissing'] = try_set(
            obj=use_missing,
            none_acceptable=True,
            is_of_type=bool)
    if min_data_per_group is not None:
        inputs['MinDataPerGroup'] = try_set(
            obj=min_data_per_group,
            none_acceptable=True,
            is_of_type=numbers.Real,
            valid_range={
                'Inf': 0,
                'Max': 2147483647})
    if max_cat_threshold is not None:
        inputs['MaxCatThreshold'] = try_set(
            obj=max_cat_threshold,
            none_acceptable=True,
            is_of_type=numbers.Real,
            valid_range={
                'Inf': 0,
                'Max': 2147483647})
    if cat_smooth is not None:
        inputs['CatSmooth'] = try_set(
            obj=cat_smooth,
            none_acceptable=True,
            is_of_type=numbers.Real, valid_range={'Min': 0.0})
    if cat_l2 is not None:
        inputs['CatL2'] = try_set(
            obj=cat_l2,
            none_acceptable=True,
            is_of_type=numbers.Real, valid_range={'Min': 0.0})
    if seed is not None:
        inputs['Seed'] = try_set(
            obj=seed,
            none_acceptable=True,
            is_of_type=numbers.Real)
    if parallel_trainer is not None:
        inputs['ParallelTrainer'] = try_set(
            obj=parallel_trainer,
            none_acceptable=True,
            is_of_type=dict)
    if predictor_model is not None:
        outputs['PredictorModel'] = try_set(
            obj=predictor_model, none_acceptable=False, is_of_type=str)

    input_variables = {
        x for x in unlist(inputs.values())
        if isinstance(x, str) and x.startswith("$")}
    output_variables = {
        x for x in unlist(outputs.values())
        if isinstance(x, str) and x.startswith("$")}

    entrypoint = EntryPoint(
        name=entrypoint_name, inputs=inputs, outputs=outputs,
        input_variables=input_variables,
        output_variables=output_variables)
    return entrypoint
