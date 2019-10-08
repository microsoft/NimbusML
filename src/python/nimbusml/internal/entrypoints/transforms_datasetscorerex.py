"""
Transforms.DatasetScorerEx
"""


from ..utils.entrypoints import EntryPoint
from ..utils.utils import try_set, unlist


def transforms_datasetscorerex(
        data,
        predictor_model,
        scored_data=None,
        scoring_transform=None,
        suffix=None,
        **params):
    """
    **Description**
        Score a dataset with a predictor model

    :param data: The dataset to be scored (inputs).
    :param predictor_model: The predictor model to apply to data
        (inputs).
    :param suffix: Suffix to append to the score columns (inputs).
    :param scored_data: The scored dataset (outputs).
    :param scoring_transform: The scoring transform (outputs).
    """

    entrypoint_name = 'Transforms.DatasetScorerEx'
    inputs = {}
    outputs = {}

    if data is not None:
        inputs['Data'] = try_set(
            obj=data,
            none_acceptable=False,
            is_of_type=str)
    if predictor_model is not None:
        inputs['PredictorModel'] = try_set(
            obj=predictor_model,
            none_acceptable=False,
            is_of_type=str)
    if suffix is not None:
        inputs['Suffix'] = try_set(
            obj=suffix,
            none_acceptable=True,
            is_of_type=str)
    if scored_data is not None:
        outputs['ScoredData'] = try_set(
            obj=scored_data,
            none_acceptable=False,
            is_of_type=str)
    if scoring_transform is not None:
        outputs['ScoringTransform'] = try_set(
            obj=scoring_transform, none_acceptable=False, is_of_type=str)

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
