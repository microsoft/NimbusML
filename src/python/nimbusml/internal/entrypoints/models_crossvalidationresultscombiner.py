# - Generated by tools/entrypoint_compiler.py: do not edit by hand
"""
Models.CrossValidationResultsCombiner
"""


from ..utils.entrypoints import EntryPoint
from ..utils.utils import try_set, unlist


def models_crossvalidationresultscombiner(
        warnings=None,
        overall_metrics=None,
        per_instance_metrics=None,
        confusion_matrix=None,
        overall_metrics=None,
        per_instance_metrics=None,
        confusion_matrix=None,
        warnings=None,
        kind='SignatureBinaryClassifierTrainer',
        label_column='Label',
        weight_column='Weight',
        group_column='GroupId',
        name_column='Name',
        **params):
    """
    **Description**
        Combine the metric data views returned from cross validation.

    :param overall_metrics: Overall metrics datasets (inputs).
    :param per_instance_metrics: Per instance metrics datasets
        (inputs).
    :param confusion_matrix: Confusion matrix datasets (inputs).
    :param warnings: Warning datasets (inputs).
    :param kind: Specifies the trainer kind, which determines the
        evaluator to be used. (inputs).
    :param label_column: The label column name (inputs).
    :param weight_column: Column to use for example weight (inputs).
    :param group_column: Column to use for grouping (inputs).
    :param name_column: Name column name (inputs).
    :param warnings: Warning dataset (outputs).
    :param overall_metrics: Overall metrics dataset (outputs).
    :param per_instance_metrics: Per instance metrics dataset
        (outputs).
    :param confusion_matrix: Confusion matrix dataset (outputs).
    """

    entrypoint_name = 'Models.CrossValidationResultsCombiner'
    inputs = {}
    outputs = {}

    if overall_metrics is not None:
        inputs['OverallMetrics'] = try_set(
            obj=overall_metrics,
            none_acceptable=True,
            is_of_type=list)
    if per_instance_metrics is not None:
        inputs['PerInstanceMetrics'] = try_set(
            obj=per_instance_metrics,
            none_acceptable=True,
            is_of_type=list)
    if confusion_matrix is not None:
        inputs['ConfusionMatrix'] = try_set(
            obj=confusion_matrix,
            none_acceptable=True,
            is_of_type=list)
    if warnings is not None:
        inputs['Warnings'] = try_set(
            obj=warnings,
            none_acceptable=True,
            is_of_type=list)
    if kind is not None:
        inputs['Kind'] = try_set(
            obj=kind,
            none_acceptable=False,
            is_of_type=str,
            values=[
                'SignatureBinaryClassifierTrainer',
                'SignatureMulticlassClassificationTrainer',
                'SignatureRankerTrainer',
                'SignatureRegressorTrainer',
                'SignatureMultiOutputRegressorTrainer',
                'SignatureAnomalyDetectorTrainer',
                'SignatureClusteringTrainer'])
    if label_column is not None:
        inputs['LabelColumn'] = try_set(
            obj=label_column,
            none_acceptable=True,
            is_of_type=str,
            is_column=True)
    if weight_column is not None:
        inputs['WeightColumn'] = try_set(
            obj=weight_column,
            none_acceptable=True,
            is_of_type=str,
            is_column=True)
    if group_column is not None:
        inputs['GroupColumn'] = try_set(
            obj=group_column,
            none_acceptable=True,
            is_of_type=str,
            is_column=True)
    if name_column is not None:
        inputs['NameColumn'] = try_set(
            obj=name_column,
            none_acceptable=True,
            is_of_type=str,
            is_column=True)
    if warnings is not None:
        outputs['Warnings'] = try_set(
            obj=warnings, none_acceptable=False, is_of_type=str)
    if overall_metrics is not None:
        outputs['OverallMetrics'] = try_set(
            obj=overall_metrics, none_acceptable=False, is_of_type=str)
    if per_instance_metrics is not None:
        outputs['PerInstanceMetrics'] = try_set(
            obj=per_instance_metrics, none_acceptable=False, is_of_type=str)
    if confusion_matrix is not None:
        outputs['ConfusionMatrix'] = try_set(
            obj=confusion_matrix, none_acceptable=False, is_of_type=str)

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
