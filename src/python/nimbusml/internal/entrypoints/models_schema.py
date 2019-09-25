"""
Models.Summarizer
"""


from ..utils.entrypoints import EntryPoint
from ..utils.utils import try_set, unlist


def models_schema(
        transform_model,
        schema=None,
        **params):
    """
    **Description**
        Retreives input/output column schema for transform model.

    :param transform_model: The transform model.
    """

    entrypoint_name = 'Models.Schema'
    inputs = {}
    outputs = {}

    if transform_model is not None:
        inputs['Model'] = try_set(
            obj=transform_model,
            none_acceptable=False,
            is_of_type=str)
    if schema is not None:
        outputs['Schema'] = try_set(
            obj=schema,
            none_acceptable=False,
            is_of_type=str)
    
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
