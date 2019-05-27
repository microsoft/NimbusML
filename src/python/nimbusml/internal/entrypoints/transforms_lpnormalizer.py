# - Generated by tools/entrypoint_compiler.py: do not edit by hand
"""
Transforms.LpNormalizer
"""


from ..utils.entrypoints import EntryPoint
from ..utils.utils import try_set, unlist


def transforms_lpnormalizer(
        column,
        data,
        output_data=None,
        model=None,
        norm='L2',
        sub_mean=False,
        **params):
    """
    **Description**
        Normalize vectors (rows) individually by rescaling them to unit norm
        (L2, L1 or LInf). Performs the following operation on a
        vector X: Y = (X - M) / D, where M is mean and D is either L2
        norm, L1 norm or LInf norm.

    :param column: New column definition(s) (optional form: name:src)
        (inputs).
    :param norm: The norm to use to normalize each sample (inputs).
    :param data: Input dataset (inputs).
    :param sub_mean: Subtract mean from each value before normalizing
        (inputs).
    :param output_data: Transformed dataset (outputs).
    :param model: Transform model (outputs).
    """

    entrypoint_name = 'Transforms.LpNormalizer'
    inputs = {}
    outputs = {}

    if column is not None:
        inputs['Column'] = try_set(
            obj=column,
            none_acceptable=False,
            is_of_type=list,
            is_column=True)
    if norm is not None:
        inputs['Norm'] = try_set(
            obj=norm,
            none_acceptable=True,
            is_of_type=str,
            values=[
                'L2',
                'StandardDeviation',
                'L1',
                'Infinity'])
    if data is not None:
        inputs['Data'] = try_set(
            obj=data,
            none_acceptable=False,
            is_of_type=str)
    if sub_mean is not None:
        inputs['SubMean'] = try_set(
            obj=sub_mean,
            none_acceptable=True,
            is_of_type=bool)
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
