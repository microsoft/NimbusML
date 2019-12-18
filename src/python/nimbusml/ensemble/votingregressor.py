# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
"""
VotingRegressor
"""

__all__ = ["VotingRegressor"]


from sklearn.base import RegressorMixin
from ..base_predictor import BasePredictor
from ..internal.utils.entrypoints import EntryPoint
from ..internal.utils.utils import trace
from ..internal.core.base_pipeline_item import DefaultSignature
from ..internal.entrypoints.models_regressionensemble import \
    models_regressionensemble


class VotingRegressor(BasePredictor,
                      RegressorMixin,
                      DefaultSignature):
    """
    **Description**
        Combine regression models into an ensemble

    :param estimators: The predictors to combine into an ensemble.
    :param model_combiner: The combiner used to combine the scores.
    """
    @trace
    def __init__(self,
                 estimators,
                 combiner='Median',
                 **params):
        BasePredictor.__init__(
            self,
            type='regressor',
            has_implicit_predictors=True,
            **params)
        self._combiner = combiner

        if not estimators:
            raise ValueError("Estimators must be specified.")
        elif not isinstance(estimators, list):
            raise ValueError("Estimators must be of type list.")
        elif any([not isinstance(e, BasePredictor) for e in estimators]):
            raise ValueError("Estimators must be predictors.")
        else:
            self._predictors = estimators

    @property
    def _entrypoint(self):
        return models_regressionensemble

    @trace
    def _get_node(self, **all_args):
        algo_args = dict(
            model_combiner=self._combiner,
            validate_pipelines=True)

        all_args.update(algo_args)
        return self._entrypoint(**all_args)

