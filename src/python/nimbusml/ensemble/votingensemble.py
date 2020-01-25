# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
"""
Voting based ensembles.
"""

__all__ = ["VotingRegressor"]

from collections import OrderedDict

from sklearn.base import RegressorMixin
from ..base_predictor import BasePredictor
from ..internal.utils.data_roles import DataRoles
from ..internal.utils.utils import trace
from ..internal.core.base_pipeline_item import DefaultSignature
from ..internal.entrypoints.transforms_manyheterogeneousmodelcombiner \
    import transforms_manyheterogeneousmodelcombiner
from ..internal.entrypoints.models_regressionensemble import \
    models_regressionensemble


class VotingEnsemble(BasePredictor,
                     DefaultSignature):
    """
    Base class for voting based ensembles 
    """
    @trace
    def __init__(self,
                 type,
                 estimators,
                 combiner='Median',
                 **params):
        BasePredictor.__init__(
            self,
            type=type,
            **params)

        self.combiner = combiner

        if not estimators:
            raise ValueError("Estimators must be specified.")
        elif not isinstance(estimators, list):
            raise ValueError("Estimators must be of type list.")
        elif any([not isinstance(e, BasePredictor) for e in estimators]):
            raise ValueError("Estimators must be predictors.")
        elif any([e.type != type for e in estimators]):
            raise ValueError("Estimators must be of type: " + type)
        else:
            self.estimators = estimators

    @trace
    def _get_node(self, **all_args):
        algo_args = dict(
            model_combiner=self.combiner,
            validate_pipelines=True)

        all_args.update(algo_args)
        return self._entrypoint(**all_args)

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
        combiner_nodes = []
        predictor_nodes = []
        predictor_models = []
        learner_features = None
        implicit_transforms = []

        for index, predictor in enumerate(self.estimators):
            suffix = '_p' + str(index)
            learner_graph_nodes, learner_features = \
                predictor._get_graph_nodes(
                        transform_nodes,
                        columns_out,
                        label_column,
                        weight_column,
                        output_data,
                        output_model,
                        predictor_model + suffix,
                        y,
                        strategy_iosklearn)

            if not implicit_transforms:
                implicit_transforms = learner_graph_nodes['implicit_nodes']

            elif learner_graph_nodes['implicit_nodes'] != implicit_transforms:
                raise RuntimeError('Predictors do not have matching'
                                   'implicit transforms.')

            predictor_node = learner_graph_nodes['learner_node'][0]
            predictor_node._implicit = True
            predictor_nodes.append(predictor_node)

            if index == 0:
                self._fit_info_proxy = predictor, predictor_node

            transform_models = []
            transform_models.extend([ep.outputs["Model"] for ep in transform_nodes])
            transform_models.extend([ep.outputs["Model"] for ep in implicit_transforms])

            predictor_model_var_name = "$predictor_model_combined" + str(index)
            predictor_models.append(predictor_model_var_name)

            combine_model_node = transforms_manyheterogeneousmodelcombiner(
                transform_models=transform_models,
                predictor_model=predictor_node.outputs['PredictorModel'],
                model=predictor_model_var_name)
            combine_model_node._implicit = True
            combiner_nodes.append(combine_model_node)

        learner_node = self._get_node(
            models=predictor_models,
            predictor_model=output_model)

        implicit_nodes = []
        implicit_nodes.extend(implicit_transforms)
        implicit_nodes.extend(predictor_nodes)
        implicit_nodes.extend(combiner_nodes)

        graph_nodes = OrderedDict([
            ('implicit_nodes', implicit_nodes),
            ('learner_node', [learner_node])
        ])

        return graph_nodes, learner_features

    def _get_fit_info_proxy(self):
        return self._fit_info_proxy

    def _move_role_info(self):
        DataRoles.move_role_info(self)

        for estimator in self.estimators:
            estimator._move_role_info()


class VotingRegressor(VotingEnsemble,
                      RegressorMixin):
    """
    **Description**
        Combine regression models into an ensemble. This class is
        modeled after the `VotingRegressor <https://scikit-learn.org/stable/modules/ensemble.html#voting-regressor>`_
        in scikit-learn.

        Note, in the majority of cases, the ``normalize``
        parameter must be explicitly specifed and must be
        the same value for all the estimators which are part
        of the ensemble.

        Here is an example::

            r1 = OnlineGradientDescentRegressor(normalize="Yes")
            r2 = OrdinaryLeastSquaresRegressor(normalize="Yes")
            vr = VotingRegressor(estimators=[r1, r2], combiner='Average')
            vr.fit(X_train, y_train)
            result = vr.predict(X_test)

    :param estimators: The predictors to combine into an ensemble.
    :param model_combiner: The combiner used to combine the scores.
        Can be either 'Median' or 'Average'.

    .. index:: models, ensemble, regression

    Example:
       .. literalinclude:: /../nimbusml/examples/VotingRegressor.py
              :language: python
    """
    @trace
    def __init__(self,
                 estimators,
                 combiner='Median',
                 **params):
        VotingEnsemble.__init__(
            self,
            type='regressor',
            estimators=estimators,
            combiner=combiner,
            **params)

    @property
    def _entrypoint(self):
        return models_regressionensemble
