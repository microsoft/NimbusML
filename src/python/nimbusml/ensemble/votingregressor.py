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
from ..internal.utils.utils import trace, try_set, unlist
from ..internal.core.base_pipeline_item import DefaultSignature


def create_entrypoint(
        models,
        predictor_model=None,
        model_combiner='Median',
        validate_pipelines=True,
        **params):
    """
    **Description**
        Combine regression models into an ensemble

    :param models: The models to combine into an ensemble (inputs).
    :param model_combiner: The combiner used to combine the scores
        (inputs).
    :param validate_pipelines: Whether to validate that all the
        pipelines are identical (inputs).
    :param predictor_model: The trained model (outputs).
    """

    entrypoint_name = 'Models.RegressionEnsemble'
    inputs = {}
    outputs = {}

    if models is not None:
        inputs['Models'] = try_set(
            obj=models,
            none_acceptable=False,
            is_of_type=list)
    if model_combiner is not None:
        inputs['ModelCombiner'] = try_set(
            obj=model_combiner,
            none_acceptable=True,
            is_of_type=str,
            values=[
                'Median',
                'Average'])
    if validate_pipelines is not None:
        inputs['ValidatePipelines'] = try_set(
            obj=validate_pipelines, none_acceptable=True, is_of_type=bool)
    if predictor_model is not None:
        outputs['PredictorModel'] = try_set(
            obj=predictor_model, none_acceptable=False, is_of_type=str)

    input_variables = {
        x for x in unlist(inputs.values())
        if isinstance(x, str) and x.startswith("$")}
    output_variables = {
        x for x in unlist(outputs.values())
        if isinstance(x, str) and x.startswith("$")}

    entrypoint = EntryPoint(
        name=entrypoint_name, inputs=inputs, outputs=outputs,
        input_variables=input_variables,
        output_variables=output_variables)
    return entrypoint


class VotingRegressor(BasePredictor,
                      RegressorMixin,
                      DefaultSignature):
    """
    **Description**
        Combine regression models into an ensemble

    :param estimators: The estimators to combine into an ensemble.
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
            is_parallel_ensemble=True,
            **params)
        self._estimators = estimators
        self._combiner = combiner

    @property
    def _entrypoint(self):
        return create_entrypoint

    @trace
    def _get_node(self, **all_args):
        algo_args = dict(
            model_combiner=self._combiner,
            validate_pipelines=False)

        all_args.update(algo_args)
        return self._entrypoint(**all_args)

