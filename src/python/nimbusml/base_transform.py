# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
"""
Base class for all the transforms.
"""

__all__ = ["BaseTransform"]

import os

from sklearn.base import BaseEstimator

from . import Pipeline
from .internal.core.base_pipeline_item import BasePipelineItem
from .internal.utils.utils import trace, compare_shape, set_shape


class BaseTransform(BaseEstimator, BasePipelineItem):
    """
    Base class for all transforms.
    """

    @trace
    def __init__(self, **params):
        BasePipelineItem.__init__(self, type='transform', **params)

    @trace
    def fit_transform(self, X, y=None, as_binary_data_stream=False,
                      **params):
        """
        Fits the transform if needed and applies it to data.

        :param X: array-like with shape=[n_samples, n_features] or else
        :py:class:`nimbusml.FileDataStream`
        :param y: array-like with shape=[n_samples]
        :return: pandas.DataFrame
        """
        pipeline = Pipeline([self])
        try:
            pipeline.fit_transform(
                X, y, as_binary_data_stream=as_binary_data_stream,
                **params)
        except RuntimeError as e:
            set_shape(self, X)
            self.model_ = pipeline.model
            raise e

        self.model_ = pipeline.model
        set_shape(self, X)
        return pipeline.data

    @trace
    def fit(self, X, y=None, **params):
        """
        Fits the transform.

        :param X: array-like with shape=[n_samples, n_features] or else
        :py:class:`nimbusml.FileDataStream`
        :param y: array-like with shape=[n_samples]
        :return: self
        """
        pipeline = Pipeline([self])
        try:
            pipeline.fit(X, y, **params)
        except RuntimeError as e:
            set_shape(self, X)
            self.model_ = pipeline.model
            raise e

        self.model_ = pipeline.model
        set_shape(self, X)
        return self

    @trace
    def transform(self, X, as_binary_data_stream=False, **params):
        """
        Applies transform to data.

        :param X: array-like with shape=[n_samples, n_features] or else
        :py:class:`nimbusml.FileDataStream`
        :return: pandas.DataFrame
        """
        # Check that the input is of the same shape as the one passed
        # during
        # fit.
        compare_shape(self, X)

        pipeline = Pipeline(model=self.model_)
        data = pipeline.transform(
            X, as_binary_data_stream=as_binary_data_stream, **params)
        return data

    @trace
    def export_to_onnx(self, *args, **kwargs):
        """
        Export the model to the ONNX format.

        See :py:meth:`nimbusml.Pipeline.export_to_onnx` for accepted arguments.
        """
        if not hasattr(self, 'model_') \
            or self.model_ is None \
            or not os.path.isfile(self.model_):

            raise ValueError("Model is not fitted. Train or load a model before "
                             "export_to_onnx().")

        pipeline = Pipeline(model=self.model_)
        pipeline.export_to_onnx(*args, **kwargs)
