"""
Transforms.VariableColumnTransform
"""


from ..utils.entrypoints import EntryPoint
from ..utils.utils import try_set, unlist


def transforms_variablecolumn(
        data,
        output_data=None,
        model=None,
        features=None,
        length_column_name=None,
        **params):
    """
    **Description**
        Combines the specified input columns in to a
        single variable length vectorized column.

    :param data: Input dataset (inputs).
    :param output_data: Transformed dataset (outputs).
    :param model: Transform model (outputs).
    """

    entrypoint_name = 'Transforms.VariableColumnTransform'
    inputs = {}
    outputs = {}

    if data is not None:
        inputs['Data'] = try_set(
            obj=data,
            none_acceptable=False,
            is_of_type=str)
    if features is not None:
        inputs['Features'] = try_set(
            obj=features,
            none_acceptable=True,
            is_of_type=list,
            is_column=True)
    if length_column_name is not None:
        inputs['LengthColumnName'] = try_set(
            obj=length_column_name,
            none_acceptable=True,
            is_of_type=str)
    if output_data is not None:
        outputs['OutputData'] = try_set(
            obj=output_data,
            none_acceptable=False,
            is_of_type=str)
    if model is not None:
        outputs['Model'] = try_set(
            obj=model,
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
