# - Generated by tools/entrypoint_compiler.py: do not edit by hand
"""
Trainers.LogisticRegressionBinaryClassifier
"""

import numbers

from ..utils.entrypoints import EntryPoint
from ..utils.utils import try_set, unlist


def trainers_logisticregressionbinaryclassifier(
        training_data,
        predictor_model=None,
        feature_column_name='Features',
        label_column_name='Label',
        example_weight_column_name=None,
        normalize_features='Auto',
        caching='Auto',
        show_training_statistics=False,
        l2_regularization=1.0,
        l1_regularization=1.0,
        optimization_tolerance=1e-07,
        history_size=20,
        enforce_non_negativity=False,
        initial_weights_diameter=0.0,
        maximum_number_of_iterations=2147483647,
        stochastic_gradient_descent_initilaization_tolerance=0.0,
        quiet=False,
        use_threads=True,
        number_of_threads=None,
        dense_optimizer=False,
        **params):
    """
    **Description**
        Logistic Regression is a method in statistics used to predict the
        probability of occurrence of an event and can be used as a
        classification algorithm. The algorithm predicts the
        probability of occurrence of an event by fitting data to a
        logistical function.

    :param training_data: The data to be used for training (inputs).
    :param feature_column_name: Column to use for features (inputs).
    :param label_column_name: Column to use for labels (inputs).
    :param example_weight_column_name: Column to use for example
        weight (inputs).
    :param normalize_features: Normalize option for the feature
        column (inputs).
    :param caching: Whether trainer should cache input training data
        (inputs).
    :param show_training_statistics: Show statistics of training
        examples. (inputs).
    :param l2_regularization: L2 regularization weight (inputs).
    :param l1_regularization: L1 regularization weight (inputs).
    :param optimization_tolerance: Tolerance parameter for
        optimization convergence. Low = slower, more accurate
        (inputs).
    :param history_size: Memory size for L-BFGS. Low=faster, less
        accurate (inputs).
    :param enforce_non_negativity: Enforce non-negative weights
        (inputs).
    :param initial_weights_diameter: Init weights diameter (inputs).
    :param maximum_number_of_iterations: Maximum iterations.
        (inputs).
    :param stochastic_gradient_descent_initilaization_tolerance: Run
        SGD to initialize LR weights, converging to this tolerance
        (inputs).
    :param quiet: If set to true, produce no output during training.
        (inputs).
    :param use_threads: Whether or not to use threads. Default is
        true (inputs).
    :param number_of_threads: Number of threads (inputs).
    :param dense_optimizer: Force densification of the internal
        optimization vectors (inputs).
    :param predictor_model: The trained model (outputs).
    """

    entrypoint_name = 'Trainers.LogisticRegressionBinaryClassifier'
    inputs = {}
    outputs = {}

    if training_data is not None:
        inputs['TrainingData'] = try_set(
            obj=training_data,
            none_acceptable=False,
            is_of_type=str)
    if feature_column_name is not None:
        inputs['FeatureColumnName'] = try_set(
            obj=feature_column_name,
            none_acceptable=True,
            is_of_type=str,
            is_column=True)
    if label_column_name is not None:
        inputs['LabelColumnName'] = try_set(
            obj=label_column_name,
            none_acceptable=True,
            is_of_type=str,
            is_column=True)
    if example_weight_column_name is not None:
        inputs['ExampleWeightColumnName'] = try_set(
            obj=example_weight_column_name,
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
    if show_training_statistics is not None:
        inputs['ShowTrainingStatistics'] = try_set(
            obj=show_training_statistics,
            none_acceptable=True,
            is_of_type=bool)
    if l2_regularization is not None:
        inputs['L2Regularization'] = try_set(
            obj=l2_regularization,
            none_acceptable=True,
            is_of_type=numbers.Real)
    if l1_regularization is not None:
        inputs['L1Regularization'] = try_set(
            obj=l1_regularization,
            none_acceptable=True,
            is_of_type=numbers.Real)
    if optimization_tolerance is not None:
        inputs['OptimizationTolerance'] = try_set(
            obj=optimization_tolerance,
            none_acceptable=True,
            is_of_type=numbers.Real)
    if history_size is not None:
        inputs['HistorySize'] = try_set(
            obj=history_size,
            none_acceptable=True,
            is_of_type=numbers.Real)
    if enforce_non_negativity is not None:
        inputs['EnforceNonNegativity'] = try_set(
            obj=enforce_non_negativity, none_acceptable=True, is_of_type=bool)
    if initial_weights_diameter is not None:
        inputs['InitialWeightsDiameter'] = try_set(
            obj=initial_weights_diameter,
            none_acceptable=True,
            is_of_type=numbers.Real)
    if maximum_number_of_iterations is not None:
        inputs['MaximumNumberOfIterations'] = try_set(
            obj=maximum_number_of_iterations,
            none_acceptable=True,
            is_of_type=numbers.Real)
    if stochastic_gradient_descent_initilaization_tolerance is not None:
        inputs['StochasticGradientDescentInitilaizationTolerance'] = try_set(
            obj=stochastic_gradient_descent_initilaization_tolerance,
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
    if number_of_threads is not None:
        inputs['NumberOfThreads'] = try_set(
            obj=number_of_threads,
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
