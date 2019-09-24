"""
Transforms.PrefixColumnConcatenator
"""


from ..utils.entrypoints import EntryPoint
from ..utils.utils import try_set, unlist


def transforms_prefixcolumnconcatenator(
        column,
        data,
        output_data=None,
        model=None,
        **params):
    """
    **Description**
        Concatenates one or more columns of the same item type by prefix.

    :param column: New column definition(s) (optional form:
        name:srcs) (inputs).
    :param data: Input dataset (inputs).
    :param output_data: Transformed dataset (outputs).
    :param model: Transform model (outputs).
    """

    entrypoint_name = 'Transforms.PrefixColumnConcatenator'
    inputs = {}
    outputs = {}

    if column is not None:
        inputs['Column'] = try_set(
            obj=column,
            none_acceptable=False,
            is_of_type=list,
            is_column=True)
    if data is not None:
        inputs['Data'] = try_set(
            obj=data,
            none_acceptable=False,
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
