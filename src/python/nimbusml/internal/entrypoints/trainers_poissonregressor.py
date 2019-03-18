# - Generated by tools/entrypoint_compiler.py: do not edit by hand
"""
Trainers.PoissonRegressor
"""

import numbers

from ..utils.entrypoints import EntryPoint
from ..utils.utils import try_set, unlist


def trainers_poissonregressor(
        training_data,
        predictor_model=None,
        feature_column='Features',
        label_column='Label',
        weight_column=None,
        normalize_features='Auto',
        caching='Auto',
        l2_weight=1.0,
        l1_weight=1.0,
        opt_tol=1e-07,
        memory_size=20,
        enforce_non_negativity=False,
        init_wts_diameter=0.0,
        max_iterations=2147483647,
        sgd_initialization_tolerance=0.0,
        quiet=False,
        use_threads=True,
        num_threads=None,
        dense_optimizer=False,
        **params):
    """
    **Description**
        Train an Poisson regression model.

    :param training_data: The data to be used for training (inputs).
    :param feature_column: Column to use for features (inputs).
    :param label_column: Column to use for labels (inputs).
    :param weight_column: Column to use for example weight (inputs).
    :param normalize_features: Normalize option for the feature
        column (inputs).
    :param caching: Whether learner should cache input training data
        (inputs).
    :param l2_weight: L2 regularization weight (inputs).
    :param l1_weight: L1 regularization weight (inputs).
    :param opt_tol: Tolerance parameter for optimization convergence.
        Low = slower, more accurate (inputs).
    :param memory_size: Memory size for L-BFGS. Low=faster, less
        accurate (inputs).
    :param enforce_non_negativity: Enforce non-negative weights
        (inputs).
    :param init_wts_diameter: Init weights diameter (inputs).
    :param max_iterations: Maximum iterations. (inputs).
    :param sgd_initialization_tolerance: Run SGD to initialize LR
        weights, converging to this tolerance (inputs).
    :param quiet: If set to true, produce no output during training.
        (inputs).
    :param use_threads: Whether or not to use threads. Default is
        true (inputs).
    :param num_threads: Number of threads (inputs).
    :param dense_optimizer: Force densification of the internal
        optimization vectors (inputs).
    :param predictor_model: The trained model (outputs).
    """

    entrypoint_name = 'Trainers.PoissonRegressor'
    inputs = {}
    outputs = {}

    if training_data is not None:
        inputs['TrainingData'] = try_set(
            obj=training_data,
            none_acceptable=False,
            is_of_type=str)
    if feature_column is not None:
        inputs['FeatureColumn'] = try_set(
            obj=feature_column,
            none_acceptable=True,
            is_of_type=str,
            is_column=True)
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
    if normalize_features is not None:
        inputs['NormalizeFeatures'] = try_set(
            obj=normalize_features,
            none_acceptable=True,
            is_of_type=str,
            values=[
                'No',
                'Warn',
                'Auto',
                'Yes'])
    if caching is not None:
        inputs['Caching'] = try_set(
            obj=caching,
            none_acceptable=True,
            is_of_type=str,
            values=[
                'Auto',
                'Memory',
                'None'])
    if l2_weight is not None:
        inputs['L2Weight'] = try_set(
            obj=l2_weight,
            none_acceptable=True,
            is_of_type=numbers.Real)
    if l1_weight is not None:
        inputs['L1Weight'] = try_set(
            obj=l1_weight,
            none_acceptable=True,
            is_of_type=numbers.Real)
    if opt_tol is not None:
        inputs['OptTol'] = try_set(
            obj=opt_tol,
            none_acceptable=True,
            is_of_type=numbers.Real)
    if memory_size is not None:
        inputs['MemorySize'] = try_set(
            obj=memory_size,
            none_acceptable=True,
            is_of_type=numbers.Real)
    if enforce_non_negativity is not None:
        inputs['EnforceNonNegativity'] = try_set(
            obj=enforce_non_negativity, none_acceptable=True, is_of_type=bool)
    if init_wts_diameter is not None:
        inputs['InitWtsDiameter'] = try_set(
            obj=init_wts_diameter,
            none_acceptable=True,
            is_of_type=numbers.Real)
    if max_iterations is not None:
        inputs['MaxIterations'] = try_set(
            obj=max_iterations,
            none_acceptable=True,
            is_of_type=numbers.Real)
    if sgd_initialization_tolerance is not None:
        inputs['SgdInitializationTolerance'] = try_set(
            obj=sgd_initialization_tolerance,
            none_acceptable=True,
            is_of_type=numbers.Real)
    if quiet is not None:
        inputs['Quiet'] = try_set(
            obj=quiet,
            none_acceptable=True,
            is_of_type=bool)
    if use_threads is not None:
        inputs['UseThreads'] = try_set(
            obj=use_threads,
            none_acceptable=True,
            is_of_type=bool)
    if num_threads is not None:
        inputs['NumThreads'] = try_set(
            obj=num_threads,
            none_acceptable=True,
            is_of_type=numbers.Real)
    if dense_optimizer is not None:
        inputs['DenseOptimizer'] = try_set(
            obj=dense_optimizer,
            none_acceptable=True,
            is_of_type=bool)
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
