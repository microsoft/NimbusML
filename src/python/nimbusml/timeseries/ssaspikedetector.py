# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
# - Generated by tools/entrypoint_compiler.py: do not edit by hand
"""
SsaSpikeDetector
"""

__all__ = ["SsaSpikeDetector"]


from sklearn.base import TransformerMixin

from ..base_transform import BaseTransform
from ..internal.core.timeseries.ssaspikedetector import \
    SsaSpikeDetector as core
from ..internal.utils.utils import trace


class SsaSpikeDetector(core, BaseTransform, TransformerMixin):
    """

    This transform detects the spikes in a seasonal time-series using
    Singular Spectrum Analysis (SSA).

    .. remarks::
        `Singular Spectrum Analysis (SSA)
        <https://en.wikipedia.org/wiki/Singular_spectrum_analysis>`_ is a
        powerful
        framework for decomposing the time-series into trend, seasonality and
        noise components as well as forecasting
        the future values of the time-series. In order to remove the effect
        of such components on anomaly detection,
        this transform adds SSA as a time-series modeler component in the
        detection pipeline.

        The SSA component will be trained and it predicts the next expected
        value on the time-series under normal condition; this expected value
        is
        further used to calculate the amount of deviation from the normal
        (predicted) behavior at that timestamp.
        The distribution of this deviation is then modeled using `Adaptive
        kernel density estimation
        <https://en.wikipedia.org/wiki/Variable_kernel_density_estimation>`_.

        The `p-value <https://en.wikipedia.org/wiki/P-value>`_ score for the
        current deviation is calculated based on the
        estimated distribution. The lower its value, the more likely the
        current point is an outlier.

    :param columns: see `Columns </nimbusml/concepts/columns>`_.

    :param training_window_size: The number of points, N, from the beginning
        of the sequence used to train the SSA
        model.

    :param confidence: The confidence for spike detection in the range [0,
        100].

    :param seasonal_window_size: An upper bound, L,  on the largest relevant
        seasonality in the input time-series, which
        also determines the order of the autoregression of SSA. It must
        satisfy 2 < L < N/2.

    :param side: The argument that determines whether to detect positive or
        negative anomalies, or both. Available
        options are {``Positive``, ``Negative``, ``TwoSided``}.

    :param pvalue_history_length: The size of the sliding window for computing
        the p-value.

    :param error_function: The function used to compute the error between the
        expected and the observed value. Possible
        values are {``SignedDifference``, ``AbsoluteDifference``,
        ``SignedProportion``, ``AbsoluteProportion``,
        ``SquaredDifference``}.

    :param params: Additional arguments sent to compute engine.

    .. seealso::
        :py:func:`IIDChangePointDetector
        <nimbusml.preprocessing.timeseries.IIDChangePointDetector>`,
        :py:func:`IIDSpikeDetector
        <nimbusml.preprocessing.timeseries.IIDSpikeDetector>`,
        :py:func:`SsaChangePointDetector
        <nimbusml.preprocessing.timeseries.SsaChangePointDetector>`.

    .. index:: models, timeseries, transform

    Example:
       .. literalinclude:: /../nimbusml/examples/examples_from_dataframe/SsaSpikeDetector_df.py
              :language: python
    """

    @trace
    def __init__(
            self,
            training_window_size=100,
            confidence=99.0,
            seasonal_window_size=10,
            side='TwoSided',
            pvalue_history_length=100,
            error_function='SignedDifference',
            columns=None,
            **params):

        if columns:
            params['columns'] = columns
        BaseTransform.__init__(self, **params)
        core.__init__(
            self,
            training_window_size=training_window_size,
            confidence=confidence,
            seasonal_window_size=seasonal_window_size,
            side=side,
            pvalue_history_length=pvalue_history_length,
            error_function=error_function,
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
