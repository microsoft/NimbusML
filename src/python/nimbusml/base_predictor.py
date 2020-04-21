# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
"""
Base class for all the predictors.
"""

__all__ = ["BasePredictor"]

import os
from collections import OrderedDict

from sklearn.base import BaseEstimator

from . import Pipeline
from .internal.core.base_pipeline_item import BasePipelineItem
from .internal.utils.data_roles import Role
from .internal.utils.data_schema import DataSchema
from .internal.utils.schema_helper import _extract_label_column
from .internal.utils.utils import trace, compare_shape, set_shape

from .internal.entrypoints.transforms_featurecombiner import \
    transforms_featurecombiner
from .internal.entrypoints.transforms_labelcolumnkeybooleanconverter \
    import transforms_labelcolumnkeybooleanconverter
from .internal.entrypoints.transforms_labeltofloatconverter import \
    transforms_labeltofloatconverter
from .internal.entrypoints.transforms_optionalcolumncreator import \
    transforms_optionalcolumncreator


class BasePredictor(BaseEstimator, BasePipelineItem):
    """
    Base class for all predictors.
    """

    @trace
    def __init__(self, **params):
        if 'type' not in params:
            raise KeyError("type must be specified")
        BasePipelineItem.__init__(self, **params)

    @trace
    def fit(self, X, y=None, **params):
        """
        Fit the model given training data

        :param X: array-like with shape=[n_samples, n_features] or else
        :py:class:`nimbusml.FileDataStream`
        :param y: array-like with shape=[n_samples]
        :return: self
        """
        # Clear cached summary since it should not
        # retain its value after a new call to fit
        if hasattr(self, 'model_summary_'):
            delattr(self, 'model_summary_')

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
        Tells if the predictor was trained.
        """
        return (hasattr(self, 'model_') and
                self.model_ and
                os.path.isfile(self.model_))

    @trace
    def _invoke_inference_method(self, method, X, **params):
        """
        Returns predictions. Can be predicted labels, probabilities
        or else decision values.
        """
        if not self._is_fitted:
            raise ValueError("Model is not fitted. "
                             "fit() must be called before {}.".format(method))

        # Check that the input is of the same shape as the one passed
        # during
        # fit.
        compare_shape(self, X)

        # if not isinstance(X, DataFrame):
        #     X = check_array(X, accept_sparse=['csr'])

        pipeline = Pipeline([self], model=self.model_)
        data = getattr(pipeline, method)(X, **params)
        return data

    @trace
    def get_feature_contributions(self, X, **params):
        return self._invoke_inference_method('get_feature_contributions',
                                             X, **params)

    @trace
    def permutation_feature_importance(self, X, **params):
        return self._invoke_inference_method('permutation_feature_importance',
                                             X, **params)

    @trace
    def predict(self, X, **params):
        """
        Predict target value

        :param X: array-like with shape=[n_samples, n_features] or else
        :py:class:`nimbusml.FileDataStream`
        :return: pandas.DataFrame, [n_samples]
        """
        data = self._predict(X, **params)
        try:
            predictions = data['PredictedLabel'] if (
                        self.type == "classifier" or self.type ==
                        "clusterer") else data['Score']
        except KeyError as e:
            raise KeyError(
                "Type: {0}, unable to find column {1} in {2}".format(
                    self.type, str(e), data.columns))
        return predictions

    @trace
    def _predict(self, X, **params):
        return self._invoke_inference_method('predict', X, **params)

    @trace
    def _check_implements_method(self, method):
        """
        Checks if this class implements method
        """
        if not hasattr(self, method):
            raise AttributeError(method + "() is not available for "
                                 + str(self.__class__))

    @trace
    def _predict_proba(self, X, **params):
        """
        Generate probabilities if method exists
        """
        self._check_implements_method('predict_proba')
        return self._invoke_inference_method('predict_proba', X, **params)

    @trace
    def _decision_function(self, X, **params):
        """
        Generate decision values if method exists
        """
        self._check_implements_method('decision_function')
        return self._invoke_inference_method('decision_function', X,
                                             **params)

    @trace
    def summary(self):
        """
        Returns model summary.
        """
        if hasattr(self, 'model_summary_') and self.model_summary_ is not None:
            return self.model_summary_

        if not hasattr(
                self, 'model_') or self.model_ is None or not \
                os.path.isfile(self.model_):
            raise ValueError(
                "Model is not fitted. Train or load a model before "
                "summary().")

        pipeline = Pipeline([self], model=self.model_)
        self.model_summary_ = pipeline.summary()
        return self.model_summary_

    def _get_implicit_transforms(
            self,
            features,
            label,
            num_transforms,
            output_data,
            output_model,
            weight=None):
        if self.type == 'regressor':
            optional_node = transforms_optionalcolumncreator(
                column=[label],
                data="$input_data" if num_transforms == 0 else
                output_data + str(num_transforms),
                output_data="$optional_data",
                model=output_model + str(num_transforms + 1))
            optional_node._implicit = True
            label_node = transforms_labeltofloatconverter(
                data="$optional_data",
                label_column=label,
                output_data="$label_data",
                model=output_model + str(num_transforms + 2))
            label_node._implicit = True
            feature_node = transforms_featurecombiner(
                data="$label_data",
                features=features,
                output_data=output_data,
                model=output_model + str(num_transforms + 3))
            feature_node._implicit = True
            implicit_nodes = [optional_node, label_node, feature_node]
        elif self.type in ('classifier', 'ranker'):
            optional_node = transforms_optionalcolumncreator(
                column=[label],
                data="$input_data" if num_transforms == 0 else
                output_data + str(num_transforms),
                output_data="$optional_data",
                model=output_model + str(num_transforms + 1))
            optional_node._implicit = True
            label_node = transforms_labelcolumnkeybooleanconverter(
                data="$optional_data",
                label_column=label,
                output_data="$label_data",
                text_key_values=False,
                model=output_model + str(num_transforms + 2))
            label_node._implicit = True
            feature_node = transforms_featurecombiner(
                data="$label_data",
                features=features,
                output_data=output_data,
                model=output_model + str(num_transforms + 3))
            feature_node._implicit = True
            implicit_nodes = [optional_node, label_node, feature_node]
        elif self.type in {'recommender', 'sequence'}:
            raise NotImplementedError("Type '{0}' is not implemented yet.".
                                      format(self.type))
        else:
            feature_node = transforms_featurecombiner(
                data="$input_data" if num_transforms == 0 else
                output_data + str(num_transforms),
                features=features,
                output_data=output_data,
                model=output_model + str(num_transforms + 1))
            feature_node._implicit = True
            implicit_nodes = [feature_node]

        return implicit_nodes

    def _get_graph_nodes(
            self,
            transform_nodes,
            columns_out,
            label_column,
            weight_column,
            output_data,
            output_model,
            predictor_model,
            y,
            strategy_iosklearn):
        graph_nodes = OrderedDict()
        num_transforms = len(transform_nodes)

        if hasattr(
                self,
                'feature_column_name') and self.feature_column_name is \
                not None:
            if isinstance(self.feature_column_name, list):
                learner_features = self.feature_column_name
                self.feature_column_name = 'Features'
            else:
                learner_features = [self.feature_column_name]
        elif strategy_iosklearn in ("previous", "accumulate"):
            if hasattr(
                    self,
                    'feature') and self.feature is not None:
                if isinstance(self.feature, list):
                    learner_features = self.feature
                else:
                    learner_features = [self.feature]
                self.feature_column_name = 'Features'
            elif isinstance(columns_out, list):
                learner_features = columns_out
                self.feature_column_name = 'Features'
            elif columns_out is None:
                learner_features = ['Features']
                self.feature_column_name = 'Features'
            else:
                learner_features = [columns_out]
                self.feature_column_name = 'Features'
        else:
            raise NotImplementedError(
                "Strategy '{0}' to handle unspecified inputs is not "
                "implemented".format(strategy_iosklearn))

        if label_column is not None or self._use_role(Role.Label):
            if getattr(self, 'label_column_name_', None):
                label_column = self.label_column_name_
            elif getattr(self, 'label_column_name', None):
                label_column = self.label_column_name
            elif label_column:
                self.label_column_name = label_column
            elif y is None:
                if label_column is None:
                    label_column = Role.Label
                self.label_column_name = label_column
            else:
                label_column = _extract_label_column(
                    self, DataSchema.read_schema(y))
                if label_column is None:
                    label_column = Role.Label
                self.label_column_name = label_column

            if y is None \
                and self._use_role(Role.Label) \
                and label_column in learner_features:
                learner_features.remove(label_column)
        else:
            self.label_column_name = None
            label_column = None

        if weight_column is not None or self._use_role(Role.Weight):
            if getattr(self, 'example_weight_column_name', None):
                weight_column = self.example_weight_column_name
            elif weight_column:
                self.example_weight_column_name = weight_column
        else:
            self.example_weight_column_name = None
            weight_column = None

        if (hasattr(self, 'row_group_column_name_')
                and self.row_group_column_name_ is not None):
            group_id_column = self.row_group_column_name_
        elif (hasattr(self, 'row_group_column_name') and
              self.row_group_column_name is not None):
            group_id_column = self.row_group_column_name
        else:
            group_id_column = None

        implicit_transforms = self._get_implicit_transforms(
            features=learner_features,
            label=label_column,
            weight=weight_column,
            num_transforms=num_transforms,
            output_data=output_data,
            output_model=output_model)
        graph_nodes['implicit_nodes'] = implicit_transforms

        self._check_roles()

        # todo: ideally all the nodes have the same name for params
        # so we dont have to distinguish if its learner or
        # transformer. We will supply input_data, output_data and
        # output_model vars. Its up to node to use supplied vars.
        learner_node = self._get_node(
            feature_column_name=learner_features,
            training_data=output_data,
            predictor_model=predictor_model,
            label_column_name=label_column,
            example_weight_column_name=weight_column,
            row_group_column_name=group_id_column)
        graph_nodes['learner_node'] = [learner_node]
        return graph_nodes, learner_features

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
