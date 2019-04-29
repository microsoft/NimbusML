# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
# - Generated by tools/entrypoint_compiler.py: do not edit by hand
"""
PoissonRegressionRegressor
"""

__all__ = ["PoissonRegressionRegressor"]


from sklearn.base import RegressorMixin

from ..base_predictor import BasePredictor
from ..internal.core.linear_model.poissonregressionregressor import \
    PoissonRegressionRegressor as core
from ..internal.utils.utils import trace


class PoissonRegressionRegressor(
        core, BasePredictor, RegressorMixin):
    """

    Train an Poisson regression model.

    .. remarks::
        `Poisson regression
        <https://en.wikipedia.org/wiki/Poisson_regression>`_ is a
        parameterized
        regression method. It assumes that the log of the conditional mean of
        the dependent variable follows a linear function of
        the dependent variables. Assuming that the dependent variable follows
        a Poisson distribution,
        the parameters of the regressor can be estimated by maximizing the
        likelihood of the obtained observations.


        **Reference**

            `Poisson regression
            <https://en.wikipedia.org/wiki/Poisson_regression>`_


    :param feature: Column to use for features.

    :param label: Column to use for labels.

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

    :param l2_regularization: L2 regularization weight.

    :param l1_regularization: L1 regularization weight.

    :param optmization_tolerance: Tolerance parameter for optimization
        convergence. Low = slower, more accurate.

    :param history_size: Memory size for L-BFGS. Low=faster, less accurate.

    :param enforce_non_negativity: Enforce non-negative weights. This flag,
        however, does not put any constraint on the bias term; that is, the
        bias term can be still a negtaive number.

    :param initial_weights_diameter: Init weights diameter.

    :param maximum_number_of_iterations: Maximum iterations.

    :param stochastic_gradient_descent_initilaization_tolerance: Run SGD to
        initialize LR weights, converging to this tolerance.

    :param quiet: If set to true, produce no output during training.

    :param use_threads: Whether or not to use threads. Default is true.

    :param number_of_threads: Number of threads.

    :param dense_optimizer: If ``True``, forces densification of the internal
        optimization vectors. If ``False``, enables the logistic regression
        optimizer use sparse or dense internal states as it finds appropriate.
        Setting ``denseOptimizer`` to ``True`` requires the internal optimizer
        to use a dense internal state, which may help alleviate load on the
        garbage collector for some varieties of larger problems.

    :param params: Additional arguments sent to compute engine.

    .. seealso::
        :py:func:`FastLinearRegressor
        <nimbusml.linear_model.FastLinearRegressor>`,
        :py:func:`OrdinaryLeastSquaresRegressor
        <nimbusml.linear_model.OrdinaryLeastSquaresRegressor>`,
        :py:func:`LightGbmRegressor <nimbusml.ensemble.LightGbmRegressor>`,
        :py:func:`FastForestRegressor <nimbusml.ensemble.FastForestRegressor>`,
        :py:func:`FastTreesRegressor <nimbusml.ensemble.FastTreesRegressor>`,
        :py:func:`GamRegressor <nimbusml.ensemble.GamRegressor>`.

    .. index:: models, regression, linear

    Example:
       .. literalinclude:: /../nimbusml/examples/PoissonRegressionRegressor.py
              :language: python
    """

    @trace
    def __init__(
            self,
            feature='Features',
            label='Label',
            weight=None,
            normalize='Auto',
            caching='Auto',
            l2_regularization=1.0,
            l1_regularization=1.0,
            optmization_tolerance=1e-07,
            history_size=20,
            enforce_non_negativity=False,
            initial_weights_diameter=0.0,
            maximum_number_of_iterations=2147483647,
            stochastic_gradient_descent_initilaization_tolerance=0.0,
            quiet=False,
            use_threads=True,
            number_of_threads=None,
            dense_optimizer=False,
            **params):

        BasePredictor.__init__(self, type='regressor', **params)
        core.__init__(
            self,
            feature=feature,
            label=label,
            weight=weight,
            normalize=normalize,
            caching=caching,
            l2_regularization=l2_regularization,
            l1_regularization=l1_regularization,
            optmization_tolerance=optmization_tolerance,
            history_size=history_size,
            enforce_non_negativity=enforce_non_negativity,
            initial_weights_diameter=initial_weights_diameter,
            maximum_number_of_iterations=maximum_number_of_iterations,
            stochastic_gradient_descent_initilaization_tolerance=stochastic_gradient_descent_initilaization_tolerance,
            quiet=quiet,
            use_threads=use_threads,
            number_of_threads=number_of_threads,
            dense_optimizer=dense_optimizer,
            **params)

    def get_params(self, deep=False):
        """
        Get the parameters for this operator.
        """
        return core.get_params(self)
