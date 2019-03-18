# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
# - Generated by tools/entrypoint_compiler.py: do not edit by hand
"""
LightGbmRegressor
"""

__all__ = ["LightGbmRegressor"]


from sklearn.base import RegressorMixin

from ..base_predictor import BasePredictor
from ..internal.core.ensemble.lightgbmregressor import \
    LightGbmRegressor as core
from ..internal.utils.utils import trace


class LightGbmRegressor(core, BasePredictor, RegressorMixin):
    """

    Gradient Boosted Decision Trees

    .. remarks::
        Light GBM is an open source implementation of boosted trees. It is
        available in nimbusml as a binary classification trainer, a multi-class
        trainer, a regression trainer and a ranking trainer.


        **Reference**

            `GitHub: LightGBM <https://github.com/Microsoft/LightGBM/wiki>`_


    :param feature: see `Columns </nimbusml/concepts/columns>`_.

    :param group_id: see `Columns </nimbusml/concepts/columns>`_.

    :param label: see `Columns </nimbusml/concepts/columns>`_.

    :param weight: see `Columns </nimbusml/concepts/columns>`_.

    :param num_boost_round: Number of iterations.

    :param learning_rate: Shrinkage rate for trees, used to prevent over-
        fitting. Range: (0,1].

    :param num_leaves: The maximum number of leaves (terminal nodes) that can
        be created in any tree. Higher values potentially increase the size of
        the tree and get better precision, but risk overfitting and requiring
        longer training times.

    :param min_data_per_leaf: Minimum number of instances needed in a child.

    :param booster: Which booster to use. Available options are:

        #. :py:func:`Dart <nimbusml.ensemble.booster.Dart>`
        #. :py:func:`Gbdt <nimbusml.ensemble.booster.Gbdt>`
        #. :py:func:`Goss <nimbusml.ensemble.booster.Goss>`.

    :param normalize: If ``Auto``, the choice to normalize depends on the
        preference declared by the algorithm. This is the default choice. If
        ``No``, no normalization is performed. If ``Yes``, normalization always
        performed. If ``Warn``, if normalization is needed by the algorithm, a
        warning message is displayed but normalization is not performed. If
        normalization is performed, a ``MaxMin`` normalizer is used. This
        normalizer preserves sparsity by mapping zero to zero.

    :param caching: Whether learner should cache input training data.

    :param max_bin: Max number of bucket bin for features.

    :param verbose_eval: Verbose.

    :param silent: Printing running messages.

    :param n_thread: Number of parallel threads used to run LightGBM.

    :param eval_metric: Evaluation metrics.

    :param use_softmax: Use softmax loss for the multi classification.

    :param early_stopping_round: Rounds of early stopping, 0 will disable it.

    :param custom_gains: Comma seperated list of gains associated to each
        relevance label.

    :param sigmoid: Parameter for the sigmoid function. Used only in
        LightGbmBinaryTrainer, LightGbmMulticlassTrainer and in
        LightGbmRankingTrainer.

    :param batch_size: Number of entries in a batch when loading data.

    :param use_cat: Enable categorical split or not.

    :param use_missing: Enable missing value auto infer or not.

    :param min_data_per_group: Min number of instances per categorical group.

    :param max_cat_threshold: Max number of categorical thresholds.

    :param cat_smooth: Lapalace smooth term in categorical feature spilt. Avoid
        the bias of small categories.

    :param cat_l2: L2 Regularization for categorical split.

    :param random_state: Sets the random seed for LightGBM to use.

    :param parallel_trainer: Parallel LightGBM Learning Algorithm.

    :param params: Additional arguments sent to compute engine.

    .. seealso::
        :py:class:`FastTreesBinaryClassifier
        <nimbusml.ensemble.FastTreesBinaryClassifier>`,
        :py:class:`FastForestRegressor <nimbusml.ensemble.FastForestRegressor>`,
        :py:func:`Dart <nimbusml.ensemble.booster.Dart>`,
        :py:func:`Goss <nimbusml.ensemble.booster.Goss>`,
        :py:func:`Gbdt <nimbusml.ensemble.booster.Gbdt>`

    .. index:: models, classification, regression

    Example:
       .. literalinclude:: /../nimbusml/examples/LightGbmRegressor.py
              :language: python
    """

    @trace
    def __init__(
            self,
            num_boost_round=100,
            learning_rate=None,
            num_leaves=None,
            min_data_per_leaf=None,
            booster=None,
            normalize='Auto',
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
            random_state=None,
            parallel_trainer=None,
            feature=None,
            group_id=None,
            label=None,
            weight=None,
            **params):

        if 'feature_column' in params:
            raise NameError(
                "'feature_column' must be renamed to 'feature'")
        if feature:
            params['feature_column'] = feature
        if 'group_id_column' in params:
            raise NameError(
                "'group_id_column' must be renamed to 'group_id'")
        if group_id:
            params['group_id_column'] = group_id
        if 'label_column' in params:
            raise NameError(
                "'label_column' must be renamed to 'label'")
        if label:
            params['label_column'] = label
        if 'weight_column' in params:
            raise NameError(
                "'weight_column' must be renamed to 'weight'")
        if weight:
            params['weight_column'] = weight
        BasePredictor.__init__(self, type='regressor', **params)
        core.__init__(
            self,
            num_boost_round=num_boost_round,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            min_data_per_leaf=min_data_per_leaf,
            booster=booster,
            normalize=normalize,
            caching=caching,
            max_bin=max_bin,
            verbose_eval=verbose_eval,
            silent=silent,
            n_thread=n_thread,
            eval_metric=eval_metric,
            use_softmax=use_softmax,
            early_stopping_round=early_stopping_round,
            custom_gains=custom_gains,
            sigmoid=sigmoid,
            batch_size=batch_size,
            use_cat=use_cat,
            use_missing=use_missing,
            min_data_per_group=min_data_per_group,
            max_cat_threshold=max_cat_threshold,
            cat_smooth=cat_smooth,
            cat_l2=cat_l2,
            random_state=random_state,
            parallel_trainer=parallel_trainer,
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
