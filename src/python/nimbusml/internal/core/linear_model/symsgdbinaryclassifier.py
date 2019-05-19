# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
# - Generated by tools/entrypoint_compiler.py: do not edit by hand
"""
SymSgdBinaryClassifier
"""

__all__ = ["SymSgdBinaryClassifier"]


from ...entrypoints.trainers_symsgdbinaryclassifier import \
    trainers_symsgdbinaryclassifier
from ...utils.utils import trace
from ..base_pipeline_item import BasePipelineItem, DefaultSignatureWithRoles


class SymSgdBinaryClassifier(
        BasePipelineItem,
        DefaultSignatureWithRoles):
    """

    Train an symbolic SGD model.

    .. remarks::
        Stochastic gradient descent (SGD) is a well known method for
        regression and classification
        tasks, and is primarily a sequential algorithm. The
        ``SymSgdBinaryClassifier`` is an
        implementation of a parallel SGD algorithm that, to a first-order
        approximation, retains the
        sequential semantics of SGD. Each thread learns a local model as well
        a `model combiner`
        which allows local models to be combined to to produce what a
        sequential model would have
        produced.



        **Reference**

            `Parallel Stochastic Gradient Descent with Sound Combiners
            <https://arxiv.org/pdf/1705.08030.pdf>`_


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

    :param number_of_iterations: Number of passes over the data.

    :param learning_rate: Learning rate.

    :param l2_regularization: L2 regularization.

    :param number_of_threads: Degree of lock-free parallelism. Determinism not
        guaranteed. Multi-threading is not supported currently.

    :param tolerance: Tolerance for difference in average loss in consecutive
        passes.

    :param update_frequency: The number of iterations each thread learns a
        local model until combining it with the global model. Low value means
        more updated global model and high value means less cache traffic.

    :param memory_size: Memory size for L-BFGS. Lower=faster, less accurate.
        The technique used for optimization here is L-BFGS, which uses only a
        limited amount of memory to compute the next step direction. This
        parameter indicates the number of past positions and gradients to store
        for the computation of the next step. Must be greater than or equal to
        ``1``.

    :param shuffle: Shuffle data?.

    :param positive_instance_weight: Apply weight to the positive class, for
        imbalanced data.

    :param params: Additional arguments sent to compute engine.

    .. seealso::
        :py:class:`LogisticRegressionBinaryClassifier
        <nimbusml.linear_model.LogisticRegressionBinaryClassifier>`,
        :py:class:`SgdBinaryClassifier
        <nimbusml.linear_model.SgdBinaryClassifier>`,
        :py:class:`FastLinearBinaryClassifier
        <nimbusml.linear_model.FastLinearBinaryClassifier>`


    .. index:: models, parallel, SGD, symbolic

    Example:
       .. literalinclude:: /../nimbusml/examples/SymSgdBinaryClassifier.py
              :language: python
    """

    @trace
    def __init__(
            self,
            normalize='Auto',
            caching='Auto',
            number_of_iterations=50,
            learning_rate=None,
            l2_regularization=0.0,
            number_of_threads=None,
            tolerance=0.0001,
            update_frequency=None,
            memory_size=1024,
            shuffle=True,
            positive_instance_weight=1.0,
            **params):
        BasePipelineItem.__init__(
            self, type='classifier', **params)

        self.normalize = normalize
        self.caching = caching
        self.number_of_iterations = number_of_iterations
        self.learning_rate = learning_rate
        self.l2_regularization = l2_regularization
        self.number_of_threads = number_of_threads
        self.tolerance = tolerance
        self.update_frequency = update_frequency
        self.memory_size = memory_size
        self.shuffle = shuffle
        self.positive_instance_weight = positive_instance_weight

    @property
    def _entrypoint(self):
        return trainers_symsgdbinaryclassifier

    @trace
    def _get_node(self, **all_args):
        algo_args = dict(
            feature_column_name=self._getattr_role(
                'feature_column_name',
                all_args),
            label_column_name=self._getattr_role(
                'label_column_name',
                all_args),
            normalize_features=self.normalize,
            caching=self.caching,
            number_of_iterations=self.number_of_iterations,
            learning_rate=self.learning_rate,
            l2_regularization=self.l2_regularization,
            number_of_threads=self.number_of_threads,
            tolerance=self.tolerance,
            update_frequency=self.update_frequency,
            memory_size=self.memory_size,
            shuffle=self.shuffle,
            positive_instance_weight=self.positive_instance_weight)

        all_args.update(algo_args)
        return self._entrypoint(**all_args)
