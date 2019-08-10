# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
# - Generated by tools/entrypoint_compiler.py: do not edit by hand
"""
TensorFlowScorer
"""

__all__ = ["TensorFlowScorer"]


from sklearn.base import TransformerMixin

from ..base_transform import BaseTransform
from ..internal.core.preprocessing.tensorflowscorer import \
    TensorFlowScorer as core
from ..internal.utils.utils import trace


class TensorFlowScorer(core, BaseTransform, TransformerMixin):
    """

    Transforms the data using the
    `TensorFlow
    <https://www.tensorflow.org/>`_ model.

    .. remarks::
       The TensorflowTransform extracts the specified outputs
       from the operations computed on the graph (given the
       input(s)) using a pre-trained
       `TensorFlow <https://www.tensorflow.org/>`_ model.
       The transform takes as input the Tensorflow model together
       with the names of the inputs to the model and names of
       the operations for which output values will be extracted
       from the model.

       This transform has the floowing assumptions regarding the
       input, output and processing of data:

        * The transform currently accepts the
            `frozen TensorFlow model
            <https://www.tensorflow.org/mobile/prepare_models>`_
            as the input.
        * The name of input column(s) should match the name
            of input(s) in Tensorflow model.
        * The name of each output column should match one of the
            operations in the Tensorflow graph.

    :param columns: see `Columns </nimbusml/concepts/columns>`_.

    :param model_location: TensorFlow model used by the transform. Please see
        https://www.tensorflow.org/mobile/prepare_models for more details.

    :param input_columns: The names of the model inputs.

    :param output_columns: The name of the outputs.

    :param batch_size: Number of samples to use for mini-batch training.

    :param add_batch_dimension_inputs: Add a batch dimension to the input e.g.
        input = [224, 224, 3] => [-1, 224, 224, 3].

    :param params: Additional arguments sent to compute engine.

    .. index:: transform

    Example:
       .. literalinclude:: /../nimbusml/examples/TensorFlowScorer.py
              :language: python
    """

    @trace
    def __init__(
            self,
            model_location,
            input_columns=None,
            output_columns=None,
            batch_size=64,
            add_batch_dimension_inputs=False,
            columns=None,
            **params):

        if columns:
            params['columns'] = columns
        if columns:
            input_columns = sum(
                list(
                    columns.values()),
                []) if isinstance(
                list(
                    columns.values())[0],
                list) else list(
                    columns.values())
        if columns:
            output_columns = list(columns.keys())
        BaseTransform.__init__(self, **params)
        core.__init__(
            self,
            model_location=model_location,
            input_columns=input_columns,
            output_columns=output_columns,
            batch_size=batch_size,
            add_batch_dimension_inputs=add_batch_dimension_inputs,
            **params)
        self._columns = columns

    def get_params(self, deep=False):
        """
        Get the parameters for this operator.
        """
        return core.get_params(self)
