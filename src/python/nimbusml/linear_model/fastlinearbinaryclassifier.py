# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
# - Generated by tools/entrypoint_compiler.py: do not edit by hand
"""
FastLinearBinaryClassifier
"""

__all__ = ["FastLinearBinaryClassifier"]


from sklearn.base import ClassifierMixin

from ..base_predictor import BasePredictor
from ..internal.core.linear_model.fastlinearbinaryclassifier import \
    FastLinearBinaryClassifier as core
from ..internal.utils.utils import trace


class FastLinearBinaryClassifier(
        core, BasePredictor, ClassifierMixin):
    """

    A Stochastic Dual Coordinate Ascent (SDCA) optimization trainer
    for linear binary classification and regression.

    .. remarks::
        ``FastLinearBinaryClassifier`` is a trainer based on the Stochastic
        Dual
        Coordinate Ascent (SDCA) method, a state-of-the-art optimization
        technique for convex objective functions. The algorithm can be scaled
        for use on large out-of-memory data sets due to a semi-asynchronized
        implementation that supports multi-threading. Convergence is
        underwritten by periodically enforcing synchronization between primal
        and dual updates in a separate thread. Several choices of loss
        functions
        are also provided. The SDCA method combines several of the best
        properties and capabilities of logistic regression and SVM
        algorithms.
        For more information on SDCA, see the citations in the reference
        section.

        Traditional optimization algorithms, such as stochastic gradient
        descent
        (SGD), optimize the empirical loss function directly. The SDCA
        chooses a
        different approach that optimizes the dual problem instead. The dual
        loss function is parameterized by per-example weights. In each
        iteration,
        when a training example from the training data set is read, the
        corresponding example weight is adjusted so that the dual loss
        function
        is optimized with respect to the current example. No learning rate is
        needed by SDCA to determine step size as is required by various
        gradient
        descent methods.

        ``FastLinearBinaryClassifier`` supports binary classification with
        three
        types of loss functions currently: Log loss, hinge loss, and smoothed
        hinge loss. Elastic net regularization can be specified by the
        ``l2_weight`` and ``l1_threshold`` parameters. Note that the
        ``l2_weight``
        has an effect on the rate of convergence. In general, the larger the
        ``l2_weight``, the faster SDCA converges.

        Note that ``FastLinearBinaryClassifier`` is a stochastic and
        streaming
        optimization algorithm. The results depends on the order of the
        training
        data. For reproducible results, it is recommended that one sets
        ``shuffle`` to ``False`` and ``train_threads`` to ``1``.


        **Reference**

            `Scaling Up Stochastic Dual Coordinate Ascent
            <https://www.microsoft.com/en-us/research/wp-
            content/uploads/2016/06/main-3.pdf>`_

            `Stochastic Dual Coordinate Ascent Methods for Regularized Loss
            Minimization <http://www.jmlr.org/papers/volume14/shalev-
            shwartz13a/shalev-shwartz13a.pdf>`_


    :param feature: see `Columns </nimbusml/concepts/columns>`_.

    :param label: see `Columns </nimbusml/concepts/columns>`_.

    :param weight: see `Columns </nimbusml/concepts/columns>`_.

    :param l2_regularization: L2 regularizer constant. By default the l2
        constant is automatically inferred based on data set.

    :param l1_threshold: L1 soft threshold (L1/L2). Note that it is easier to
        control and sweep using the threshold parameter than the raw
        L1-regularizer constant. By default the l1 threshold is automatically
        inferred based on data set.

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

    :param loss: The default is :py:class:`'log' <nimbusml.loss.Log>`. Other
        choices are :py:class:`'hinge' <nimbusml.loss.Hinge>`, and
        :py:class:`'smoothed_hinge' <nimbusml.loss.SmoothedHinge>`. For more
        information, please see the documentation page about losses,
        [Loss](xref:nimbusml.loss).

    :param number_of_threads: Degree of lock-free parallelism. Defaults to
        automatic. Determinism not guaranteed.

    :param positive_instance_weight: Apply weight to the positive class, for
        imbalanced data.

    :param convergence_tolerance: The tolerance for the ratio between duality
        gap and primal loss for convergence checking.

    :param maximum_number_of_iterations: Maximum number of iterations; set to 1
        to simulate online learning. Defaults to automatic.

    :param shuffle: Shuffle data every epoch?.

    :param convergence_check_frequency: Convergence check frequency (in terms
        of number of iterations). Set as negative or zero for not checking at
        all. If left blank, it defaults to check after every 'numThreads'
        iterations.

    :param bias_learning_rate: The learning rate for adjusting bias from being
        regularized.

    :param params: Additional arguments sent to compute engine.

    .. seealso::
        :py:class:`FastLinearRegressor
        <nimbusml.linear_model.FastLinearRegressor>`,
        :py:class:`FastLinearClassifier
        <nimbusml.linear_model.FastLinearClassifier>`


    .. index:: models, linear, SDCA, stochastic, classification, regression

    Example:
       .. literalinclude:: /../nimbusml/examples/FastLinearBinaryClassifier.py
              :language: python
    """

    @trace
    def __init__(
            self,
            l2_regularization=None,
            l1_threshold=None,
            normalize='Auto',
            caching='Auto',
            loss='log',
            number_of_threads=None,
            positive_instance_weight=1.0,
            convergence_tolerance=0.1,
            maximum_number_of_iterations=None,
            shuffle=True,
            convergence_check_frequency=None,
            bias_learning_rate=0.0,
            feature=None,
            label=None,
            weight=None,
            **params):

        if 'feature_column_name' in params:
            raise NameError(
                "'feature_column_name' must be renamed to 'feature'")
        if feature:
            params['feature_column_name'] = feature
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
        BasePredictor.__init__(self, type='classifier', **params)
        core.__init__(
            self,
            l2_regularization=l2_regularization,
            l1_threshold=l1_threshold,
            normalize=normalize,
            caching=caching,
            loss=loss,
            number_of_threads=number_of_threads,
            positive_instance_weight=positive_instance_weight,
            convergence_tolerance=convergence_tolerance,
            maximum_number_of_iterations=maximum_number_of_iterations,
            shuffle=shuffle,
            convergence_check_frequency=convergence_check_frequency,
            bias_learning_rate=bias_learning_rate,
            **params)
        self.feature = feature
        self.label = label
        self.weight = weight

    @trace
    def predict_proba(self, X, **params):
        '''
        Returns probabilities
        '''
        return self._predict_proba(X, **params)

    @trace
    def decision_function(self, X, **params):
        '''
        Returns score values
        '''
        return self._decision_function(X, **params)

    def get_params(self, deep=False):
        """
        Get the parameters for this operator.
        """
        return core.get_params(self)
