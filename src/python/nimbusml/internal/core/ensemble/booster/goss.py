# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
# - Generated by tools/entrypoint_compiler.py: do not edit by hand
"""
Goss
"""

__all__ = ["Goss"]

import numbers

from ....utils.entrypoints import Component
from ....utils.utils import trace, try_set


class Goss(Component):
    """

    Gradient-based One-Side Sampling.

    .. remarks::
        Gradient-based One-Side Sampling (GOSS) employs an adaptive sampling
        named gradient-based
    sampling. For datasets with large sample size, GOSS has considerable
        advantage in terms of
    statistical and computational efficiency.



        **Reference**

            `https://papers.nips.cc/paper/6579-gradient-based-sampling-an-
            adaptive-importance-sampling-for-least-squares.pdf
            <https://papers.nips.cc/paper/6579-gradient-based-sampling-an-
            adaptive-importance-sampling-for-least-squares.pdf>`_


    :param top_rate: Retain ratio for large gradient instances.

    :param other_rate: Retain ratio for small gradient instances.

    :param minimum_split_gain: Minimum loss reduction required to make a
        further partition on a leaf node of the tree. the larger, the more
        conservative the algorithm will be.

    :param maximum_tree_depth: Maximum depth of a tree. 0 means no limit.
        However, tree still grows by best-first.

    :param minimum_child_weight: Minimum sum of instance weight(hessian) needed
        in a child. If the tree partition step results in a leaf node with the
        sum of instance weight less than min_child_weight, then the building
        process will give up further partitioning. In linear regression mode,
        this simply corresponds to minimum number of instances needed to be in
        each node. The larger, the more conservative the algorithm will be.

    :param subsample_frequency: Subsample frequency for bagging. 0 means no
        subsample. Specifies the frequency at which the bagging occurs, where
        if this is set to N, the subsampling will happen at every N
        iterations.This must be set with Subsample as this specifies the amount
        to subsample.

    :param subsample_fraction: Subsample ratio of the training instance.
        Setting it to 0.5 means that LightGBM randomly collected half of the
        data instances to grow trees and this will prevent overfitting. Range:
        (0,1].

    :param feature_fraction: Subsample ratio of columns when constructing each
        tree. Range: (0,1].

    :param l2_regularization: L2 regularization term on weights, increasing
        this value will make model more conservative.

    :param l1_regularization: L1 regularization term on weights, increase this
        value will make model more conservative.

    :param params: Additional arguments sent to compute engine.

    .. seealso::
        :py:func:`Gbdt <nimbusml.ensemble.booster.Gbdt>`,
        :py:func:`Dart <nimbusml.ensemble.booster.Dart>`,
        :py:func:`LightGbmBinaryClassifier
        <nimbusml.ensemble.LightGbmBinaryClassifier>`,
        :py:func:`LightGbmClassifier <nimbusml.ensemble.LightGbmClassifier>`,
        :py:func:`LightGbmRanker <nimbusml.ensemble.LightGbmRanker>`,
        :py:func:`LightGbmRegressor <nimbusml.ensemble.LightGbmRegressor>`

    .. index:: ensemble, booster

    Example:
       .. literalinclude:: /../nimbusml/examples/LightGbmBinaryClassifier.py
              :language: python
    """

    @trace
    def __init__(
            self,
            top_rate=0.2,
            other_rate=0.1,
            minimum_split_gain=0.0,
            maximum_tree_depth=0,
            minimum_child_weight=0.1,
            subsample_frequency=0,
            subsample_fraction=1.0,
            feature_fraction=1.0,
            l2_regularization=0.01,
            l1_regularization=0.0,
            **params):

        self.top_rate = top_rate
        self.other_rate = other_rate
        self.minimum_split_gain = minimum_split_gain
        self.maximum_tree_depth = maximum_tree_depth
        self.minimum_child_weight = minimum_child_weight
        self.subsample_frequency = subsample_frequency
        self.subsample_fraction = subsample_fraction
        self.feature_fraction = feature_fraction
        self.l2_regularization = l2_regularization
        self.l1_regularization = l1_regularization
        self.kind = 'BoosterParameterFunction'
        self.name = 'goss'
        self.settings = {}

        if top_rate is not None:
            self.settings['TopRate'] = try_set(
                obj=top_rate,
                none_acceptable=True,
                is_of_type=numbers.Real,
                valid_range={
                    'Inf': 0.0,
                    'Max': 1.0})
        if other_rate is not None:
            self.settings['OtherRate'] = try_set(
                obj=other_rate,
                none_acceptable=True,
                is_of_type=numbers.Real,
                valid_range={
                    'Inf': 0.0,
                    'Max': 1.0})
        if minimum_split_gain is not None:
            self.settings['MinimumSplitGain'] = try_set(
                obj=minimum_split_gain,
                none_acceptable=True,
                is_of_type=numbers.Real, valid_range={'Min': 0.0})
        if maximum_tree_depth is not None:
            self.settings['MaximumTreeDepth'] = try_set(
                obj=maximum_tree_depth,
                none_acceptable=True,
                is_of_type=numbers.Real,
                valid_range={
                    'Max': 2147483647,
                    'Min': 0})
        if minimum_child_weight is not None:
            self.settings['MinimumChildWeight'] = try_set(
                obj=minimum_child_weight,
                none_acceptable=True,
                is_of_type=numbers.Real, valid_range={'Min': 0.0})
        if subsample_frequency is not None:
            self.settings['SubsampleFrequency'] = try_set(
                obj=subsample_frequency,
                none_acceptable=True,
                is_of_type=numbers.Real,
                valid_range={
                    'Max': 2147483647,
                    'Min': 0})
        if subsample_fraction is not None:
            self.settings['SubsampleFraction'] = try_set(
                obj=subsample_fraction,
                none_acceptable=True,
                is_of_type=numbers.Real,
                valid_range={
                    'Inf': 0.0,
                    'Max': 1.0})
        if feature_fraction is not None:
            self.settings['FeatureFraction'] = try_set(
                obj=feature_fraction,
                none_acceptable=True,
                is_of_type=numbers.Real,
                valid_range={
                    'Inf': 0.0,
                    'Max': 1.0})
        if l2_regularization is not None:
            self.settings['L2Regularization'] = try_set(
                obj=l2_regularization,
                none_acceptable=True,
                is_of_type=numbers.Real, valid_range={'Min': 0.0})
        if l1_regularization is not None:
            self.settings['L1Regularization'] = try_set(
                obj=l1_regularization,
                none_acceptable=True,
                is_of_type=numbers.Real, valid_range={'Min': 0.0})

        super(
            Goss,
            self).__init__(
            name=self.name,
            settings=self.settings,
            kind=self.kind)
