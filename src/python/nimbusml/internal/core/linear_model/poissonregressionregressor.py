# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
# - Generated by tools/entrypoint_compiler.py: do not edit by hand
"""
PoissonRegressionRegressor
"""

__all__ = ["PoissonRegressionRegressor"]


from ...entrypoints.trainers_poissonregressor import trainers_poissonregressor
from ...utils.utils import trace
from ..base_pipeline_item import BasePipelineItem, DefaultSignatureWithRoles


class PoissonRegressionRegressor(
        BasePipelineItem,
        DefaultSignatureWithRoles):
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

    :param optimization_tolerance: Tolerance parameter for optimization
        convergence. Low = slower, more accurate.

    :param history_size: Memory size for L-BFGS. Lower=faster, less accurate.
        The technique used for optimization here is L-BFGS, which uses only a
        limited amount of memory to compute the next step direction. This
        parameter indicates the number of past positions and gradients to store
        for the computation of the next step. Must be greater than or equal to
        ``1``.

    :param enforce_non_negativity: Enforce non-negative weights. This flag,
        however, does not put any constraint on the bias term; that is, the
        bias term can be still a negtaive number.

    :param initial_weights_diameter: Sets the initial weights diameter that
        specifies the range from which values are drawn for the initial
        weights. These weights are initialized randomly from within this range.
        For example, if the diameter is specified to be ``d``, then the weights
        are uniformly distributed between ``-d/2`` and ``d/2``. The default
        value is ``0``, which specifies that all the  weights are set to zero.

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
            normalize='Auto',
            caching='Auto',
            l2_regularization=1.0,
            l1_regularization=1.0,
            optimization_tolerance=1e-07,
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
        BasePipelineItem.__init__(
            self, type='regressor', **params)

        self.normalize = normalize
        self.caching = caching
        self.l2_regularization = l2_regularization
        self.l1_regularization = l1_regularization
        self.optimization_tolerance = optimization_tolerance
        self.history_size = history_size
        self.enforce_non_negativity = enforce_non_negativity
        self.initial_weights_diameter = initial_weights_diameter
        self.maximum_number_of_iterations = maximum_number_of_iterations
        self.stochastic_gradient_descent_initilaization_tolerance = stochastic_gradient_descent_initilaization_tolerance
        self.quiet = quiet
        self.use_threads = use_threads
        self.number_of_threads = number_of_threads
        self.dense_optimizer = dense_optimizer

    @property
    def _entrypoint(self):
        return trainers_poissonregressor

    @trace
    def _get_node(self, **all_args):
        algo_args = dict(
            feature_column_name=self._getattr_role('feature_column_name', all_args),
            label_column_name=self._getattr_role('label_column_name', all_args),
            example_weight_column_name=self._getattr_role('example_weight_column_name', all_args),
            normalize_features=self.normalize,
            caching=self.caching,
            l2_regularization=self.l2_regularization,
            l1_regularization=self.l1_regularization,
            optimization_tolerance=self.optimization_tolerance,
            history_size=self.history_size,
            enforce_non_negativity=self.enforce_non_negativity,
            initial_weights_diameter=self.initial_weights_diameter,
            maximum_number_of_iterations=self.maximum_number_of_iterations,
            stochastic_gradient_descent_initilaization_tolerance=self.stochastic_gradient_descent_initilaization_tolerance,
            quiet=self.quiet,
            use_threads=self.use_threads,
            number_of_threads=self.number_of_threads,
            dense_optimizer=self.dense_optimizer)

        all_args.update(algo_args)
        return self._entrypoint(**all_args)
