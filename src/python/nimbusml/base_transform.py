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
        :param as_binary_data_stream: If ``True`` then output an IDV file.
            See `here <https://github.com/dotnet/machinelearning/blob/master/docs/code/IDataViewImplementation.md>`_
            for more information.
        :param params: Additional arguments.
            If ``as_csr=True`` and ``as_binary_data_stream=False`` then
            return the transformed data in CSR (sparse matrix) format.
            If ``as_binary_data_stream`` is also true then that
            parameter takes precedence over ``as_csr`` and the output will
            be an IDV file.

        :return: Returns a pandas DataFrame if no other output format
            is specified. See ``as_binary_data_stream`` and ``as_csr``
            for other available output formats.
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

    @property
    def _is_fitted(self):
        """
        Tells if the transform was trained.
        """
        return (hasattr(self, 'model_') and
                self.model_ and
                os.path.isfile(self.model_))

    @trace
    def transform(self, X, as_binary_data_stream=False, **params):
        """
        Applies transform to data.

        :param X: array-like with shape=[n_samples, n_features] or else
            :py:class:`nimbusml.FileDataStream`
        :param as_binary_data_stream: If ``True`` then output an IDV file.
            See `here <https://github.com/dotnet/machinelearning/blob/master/docs/code/IDataViewImplementation.md>`_
            for more information.
        :param params: Additional arguments.
            If ``as_csr=True`` and ``as_binary_data_stream=False`` then
            return the transformed data in CSR (sparse matrix) format.
            If ``as_binary_data_stream`` is also true then that
            parameter takes precedence over ``as_csr`` and the output will
            be an IDV file.

        :return: Returns a pandas DataFrame if no other output format
            is specified. See ``as_binary_data_stream`` and ``as_csr``
            for other available output formats.
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

        pipeline = Pipeline([self], model=self.model_)
        pipeline.export_to_onnx(*args, **kwargs)
