# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
# - Generated by tools/entrypoint_compiler.py: do not edit by hand
"""
IidSpikeDetector
"""

__all__ = ["IidSpikeDetector"]


from sklearn.base import TransformerMixin

from ..base_transform import BaseTransform
from ..internal.core.timeseries._iidspikedetector import \
    IidSpikeDetector as core
from ..internal.utils.utils import trace


class IidSpikeDetector(core, BaseTransform, TransformerMixin):
    """

    This transform detects the spikes in a i.i.d. sequence using adaptive
    kernel density estimation.

    .. remarks::
        ``IIDSpikeDetector`` assumes a sequence of data points that are
        independently sampled from one stationary
        distribution. `Adaptive kernel density estimation
        <https://en.wikipedia.org/wiki/Variable_kernel_density_estimation>`_
        is used to model the distribution.
        The `p-value <https://en.wikipedia.org/wiki/P-value>`_ score
        indicates the likelihood of the current observation according to
        the estimated distribution. The lower its value, the more likely the
        current point is an outlier.

    :param columns: see `Columns </nimbusml/concepts/columns>`_.

    :param confidence: The confidence for spike detection in the range [0,
        100].

    :param side: The argument that determines whether to detect positive or
        negative anomalies, or both. Available options are {``Positive``,
        ``Negative``, ``TwoSided``}.

    :param pvalue_history_length: The size of the sliding window for computing
        the p-value.

    :param params: Additional arguments sent to compute engine.

    .. seealso::
        :py:func:`IIDChangePointDetector
        <nimbusml.preprocessing.timeseries.IIDChangePointDetector>`,
        :py:func:`SsaSpikeDetector
        <nimbusml.preprocessing.timeseries.SsaSpikeDetector>`,
        :py:func:`SsaChangePointDetector
        <nimbusml.preprocessing.timeseries.SsaChangePointDetector>`.

    .. index:: models, timeseries, transform

    Example:
       .. literalinclude:: /../nimbusml/examples/examples_from_dataframe/IidSpikeDetector_df.py
              :language: python
    """

    @trace
    def __init__(
            self,
            confidence=99.0,
            side='TwoSided',
            pvalue_history_length=100,
            columns=None,
            **params):

        if columns:
            params['columns'] = columns
        BaseTransform.__init__(self, **params)
        core.__init__(
            self,
            confidence=confidence,
            side=side,
            pvalue_history_length=pvalue_history_length,
            **params)
        self._columns = columns

    def get_params(self, deep=False):
        """
        Get the parameters for this operator.
        """
        return core.get_params(self)

    def _nodes_with_presteps(self):
        """
        Inserts preprocessing before this one.
        """
        from ..preprocessing.schema import TypeConverter
        return [
            TypeConverter(
                result_type='R4')._steal_io(self),
            self]
