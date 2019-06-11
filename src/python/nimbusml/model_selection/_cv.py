# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import inspect
import time
import six

from pandas import DataFrame

from .. import Pipeline, FileDataStream
from ..internal.entrypoints.models_crossvalidator import \
    models_crossvalidator
from ..internal.entrypoints.transforms_manyheterogeneousmodelcombiner \
    import \
    transforms_manyheterogeneousmodelcombiner
from ..internal.entrypoints.transforms_modelcombiner import \
    transforms_modelcombiner
from ..internal.utils.entrypoints import Graph, GraphOutputType


# Extension method for extending a list of steps, with chaining
class _PipelineSteps(list):
    @staticmethod
    def get_last(items_list, default_value=None):
        try:
            last = items_list[-1]
        except IndexError:
            last = default_value

        return last

    def add(self, new_steps):
        if new_steps:
            if isinstance(new_steps, list):
                self.extend(new_steps)
            else:
                self.append(new_steps)
        return self

    def _get_values(self, attr_key, inner_keys):
        values = []
        keys = []
        nodes = []
        for node in self:
            attr = getattr(node, attr_key)
            for key in inner_keys:
                if key in attr:
                    keys.append(key)
                    values.append(attr[key])
                    nodes.append(node)
                    # Only key the first key
                    break
        return values, keys, nodes

    @property
    def first_input_data(self):
        # Return the first node that contain input 'Data' (for
        # transforms) or
        # 'TrainingData' (for learner)
        _, input_names, input_nodes = self._get_values(
            'inputs', ['Data', 'TrainingData'])
        return input_nodes[0], input_names[0]

    @property
    def all_output_models(self):
        models, _, _ = self._get_values(
            'outputs', ['Model', 'PredictorModel', 'OutputModel'])
        return models

    @property
    def last_output_model(self):
        return _PipelineSteps.get_last(self.all_output_models)

    @property
    def last_output_data(self):
        data_outputs, _, _ = self._get_values(
            'outputs', ['Data', 'OutputData'])
        return _PipelineSteps.get_last(data_outputs)


class CV:
    '''

    Cross Validation

    .. remarks::
        Cross Validation is a technique used for training and testing a
        model when there is only one dataset.
        The dataset is partitioned into k parts (k is specified by the
        user) called folds. Each fold, in turn,
        is used as a test set, where the rest of the data is used as a
        training set. The result is k separate models.
        The metrics for each model are reported separately, and so is
        the average of each metric on all models.

    :param pipeline: Pipeline object or a list of pipeline steps that's
    used for cross validation
    '''

    fold_column_name = 'Fold'

    def __init__(self, pipeline):
        if isinstance(pipeline, list):
            pipeline = Pipeline(pipeline)

        self._pipeline = pipeline
        self._results = None
        self._raw_results = None
        self._set_task_dependend_properties()

    def _set_task_dependend_properties(self):
        self.outputs = {
            'per_instance_metrics': '',
            'predictor_model': '',
            'overall_metrics': '',
            'warnings': ''}

        self.output_types = {
            'per_instance_metrics': GraphOutputType.BridgeReturnValue,
            'predictor_model': GraphOutputType.ModelArrayFile,
            'overall_metrics': GraphOutputType.TempFile,
            'warnings': GraphOutputType.TempFile}

        def _add_confusion_matrix():
            self.outputs['confusion_matrix'] = ''
            self.output_types[
                'confusion_matrix'] = GraphOutputType.TempFile

        def _clean_multiclass_metrics(metrics):
            cols = metrics.columns
            metrics.columns = [
                'Log-loss ' +
                c.strip('()') if 'class' in c else c for c in cols]

        def _clean_ranking_metrics(metrics):
            cols = [c for c in metrics.columns if c.startswith('@')]
            n = len(cols)
            clean_names = {}
            for i in range(n):
                c = cols[i]
                if i < n / 2:
                    prefix = 'NDCG'
                    clean_c = prefix + c
                else:
                    prefix = 'DCG'
                    # Remove .* suffix added by pandas
                    clean_c = prefix + c.split('.')[0]

                clean_names[c] = clean_c

            return metrics.rename(clean_names, axis=1)

        learner_type = self._learner_type = \
            self._pipeline._last_node_type()
        self._extra_predictions_columns = lambda _: []  # Default is empty
        self._clean_metrics_columns = lambda _: _  # Default is no-change

        if learner_type == 'regressor':
            self._cv_kind = 'SignatureRegressorTrainer'
            self._predictions_columns = [
                CV.fold_column_name,
                'Instance',
                'Label',
                'Score',
                'L1-loss',
                'L2-loss']

        elif learner_type == 'binary':
            self._cv_kind = 'SignatureBinaryClassifierTrainer'
            self._predictions_columns = [
                CV.fold_column_name,
                'Instance',
                'Label',
                'Score',
                'Probability',
                'Log-loss',
                'Assigned']
            _add_confusion_matrix()

        elif learner_type == 'multiclass':
            self._cv_kind = 'SignatureMulticlassClassificationTrainer'
            self._predictions_columns = [
                CV.fold_column_name,
                'Instance',
                'Label',
                'Log-loss',
                'Assigned']
            self._extra_predictions_columns = lambda prediction_columns: [
                c for c in prediction_columns if
                ('Score' in c or 'Class' in c)]
            self._clean_metrics_columns = _clean_multiclass_metrics
            _add_confusion_matrix()

        elif learner_type == 'ranker':
            self._cv_kind = 'SignatureRankerTrainer'
            self._predictions_columns = [
                CV.fold_column_name,
                'Instance',
                'GroupId',
                'Label',
                'Score',
                'NDCG@1',
                'NDCG@2',
                'NDCG@3',
                'DCG@1',
                'DCG@2',
                'DCG@3',
                'MaxDCG@1']
            self._clean_metrics_columns = _clean_ranking_metrics

        elif learner_type == 'clusterer':
            self._cv_kind = 'SignatureClusteringTrainer'
            self._predictions_columns = [
                CV.fold_column_name, 'Instance', 'ClusterId']
            self._extra_predictions_columns = lambda prediction_columns: [
                c for c in prediction_columns if
                ('Score' in c or 'Sorted' in c)]

        elif learner_type == 'transform':
            raise ValueError(
                'The input pipeline doesn\'t end with any predictor, '
                'which is required by CV.')

        else:
            error_msg = 'The input pipeline ends with {}, which is not ' \
                        'supported by CV.'.format(learner_type)
            raise ValueError(error_msg)

    def _get_output_name(self, name):
        if name in self.outputs:
            return '$' + name
        else:
            return None

    def _cleanup_predictions(self, predictions):
        # Predictions include features columns as well. Only keep
        # non-features
        # columns
        columns = [
            c for c in self._predictions_columns if
            c in predictions.columns]
        columns.extend(
            self._extra_predictions_columns(predictions.columns))
        return predictions[columns]

    def _bring_fold_column_to_front(self, df):
        df = df.rename(columns={'Fold Index': CV.fold_column_name})
        columns = list(df.columns.values)
        columns.insert(0, columns.pop(columns.index(CV.fold_column_name)))
        return df[columns]

    def _check_warnings(self, warnings):
        # Check warnings for missing classes warning (only applicable to
        # classification)
        if self._learner_type not in {'binary', 'multiclass'}:
            return

        warnings_text = warnings['WarningText'].values
        unseen_classes_warning = any(
            ['unlabeled instances during testing' in w.lower() for w in
             warnings_text])
        if unseen_classes_warning:
            msg = 'During cross validation, some of folds didn\'t ' \
                  'contain all classes. ' + \
                  'As a result, the calculated metrics are incorrect. To ' \
                  '' \
                  'fix the problem try:\n' + \
                  '1) Increase the size of your data\n' + \
                  '2) Apply a ToKey transform to the label column as a ' \
                  'pre-split transform\n'
            raise Warning(msg)

    def _cleanup_results(self, results, cv):
        self._check_warnings(results['warnings'])

        clean_results = {}

        # Return the output model path for each fold
        models_pattern = results['predictor_model']
        models_df = DataFrame(columns=[CV.fold_column_name, 'model'])
        for fold in range(cv):
            model_path = models_pattern.format(fold)
            models_df.loc[fold] = [fold, model_path]
        clean_results['models'] = models_df

        # Remove non-metric columns from predictions
        predictions = self._bring_fold_column_to_front(
            results['per_instance_metrics'])
        clean_results['predictions'] = self._cleanup_predictions(
            predictions)

        # Metrics
        # Firt two rows are metrics summary. The rest are per fold results.
        metrics = self._bring_fold_column_to_front(
            results['overall_metrics'])
        clean_results['metrics'] = metrics.iloc[2:, :].reset_index()
        self._clean_metrics_columns(clean_results['metrics'])
        clean_results['metrics_summary'] = \
            metrics.iloc[:2, :].set_index(CV.fold_column_name)

        if 'confusion_matrix' in results:
            clean_results[
                'confusion_matrix'] = self._bring_fold_column_to_front(
                results['confusion_matrix'])

        return clean_results

    def _process_split_start(self, split_start):
        nodes = self._pipeline.nodes
        pipeline_len = len(nodes)
        if isinstance(split_start, str):
            if split_start == 'before_transforms':
                split_index = 0
            elif split_start == 'after_transforms':
                split_index = pipeline_len - 1
            else:
                raise ValueError(
                    'String value for split_start should be either '
                    '"before_transforms" or "after_transforms"')

        if isinstance(split_start, six.integer_types):
            try:
                nodes[split_start]
            except IndexError:
                raise ValueError(
                    'Pipeline doesn\'t contain a step for split_start={'
                    '}'.format(
                        split_start))

            split_index = split_start

        # Negative indices are relative to the size of the list.
        # Convert split_index to positive number, so that it can index into
        # list of transfroms without the learner.
        if split_index < 0:
            split_index = split_index + pipeline_len

        return split_index

    def fit(
            self,
            X,
            y=None,
            cv=2,
            groups=None,
            split_start='before_transforms',
            **params):
        '''
        Cross validate the pipeline and return the results.

        :param X: The data to fit. Can be any data format that's
        acceptable by the input pipeline.

        :param y: Target value. Could be None, if X is a FileDataStream.

        :param cv: integer specifying number of folds.

        :param groups: Name of the column that contains groups.
        Sometimes there is a need to specify which examples
            should not be separated into different folds. Take,
            for instance, the ranking problem, where instances have
            a "query" and a "url" feature. Instances that have the same
            query value should always be in the same fold
            (otherwise the algorithm "cheats" by seeing examples for
            same query). In such cases, the groups column can
            be used. Data rows that have the same value for groups
            column, will be in the same fold.

        :param split_start: int, 'before_transforms',
        or 'after_transforms'. When the pipeline has many transforms, it
            would be more efficient to do the transforms before
            splitting the data, so that the the transforms run only
            once, instead of once per fold. However, with some
            transforms that learn from data, this could cause data
            leak, so extra care must be taken when using this option.
            split_start can precisely specify where data
            splitting happens:

            * 'before_transforms' means split data before all the
            transforms in the pipeline. This is the default
              behavior and would not cause any data leak.

            * 'after_transforms' means split data after all the
            transforms in the pipeline. This is the fastest option,
              but could cause data leak, depending on the transforms.
              The results from this option can be compared
              to 'before_transforms' results to ensure data leak doesn't
              happen.

            * For precise control, split_start can be specified as an
            int, which means pipeline_steps[:split_start] will
              be applied before the split, and pipeline_steps[
              split_start:] will be applied after the split. Note that
              'after_transforms' is equivalent to -1,
              and 'before_transforms' is equivalent to 0.

        :param params: Additional arguments sent to compute engine.

        :return: dict of pandas dataframes. The possible keys for this
        dict are:

            * ``'predictions'``: dataframe containing the predictions
            for input data. The prediction for each data
              point corresponds to the prediction when the fold
              containing that data point was used as test data.

            * ``'models'``: dataframe containing the model file path per
            fold.

            * ``'metrics'``: dataframe containing the metrics per fold.

            * ``'metrics_summary'``: dataframe containing the summary
            statistics of metrics.

            * ``'confusion_matrix'``: dataframe containing the confusion
            matrix per fold (only applicable to
              classification).

        Example:
           .. literalinclude:: /../nimbusml/examples/CV.py
                  :language: python
        '''

        self._results = None
        self._raw_results = None
        verbose = 1

        # _fit_graph() seems to have side-effects on the pipeline object
        # Use a clone, so that we can reuse CV object for multiple calls to
        # fit()
        pipeline = self._pipeline.clone()

        # pipeline code is convoluted. Changing _fit_graph was the
        # safest way to get the info
        # that CV requires out of Pipeline. So a lot of return values are
        # unused.
        _, X, y, weights, _, _, _, _, cv_aux_info, \
            _ = self._pipeline._fit_graph(X, y, verbose, **params)
        assert (cv_aux_info.predictor_model == '$predictor_model')
        learner_roles = pipeline._get_last_node_roles(X, y)
        group_id = learner_roles.get('GroupId')
        label_column = cv_aux_info.label_column or learner_roles.get(
            'Label') or 'Label'

        # If group_id exists (ranking) and no groups is provided,
        # use group_id for groups
        # This generates issue if group_id doesn't exist in the origin
        # data!
        # Need to infer from group_id, bug 284886
        groups = groups or group_id
        if groups is not None:
            if isinstance(X, FileDataStream):
                if groups not in cv_aux_info[0]['data_import'][0].inputs[
                        'CustomSchema']:
                    raise Exception(
                        'Default stratification column: ' +
                        str(groups) +
                        ' cannot be found in the origin data, please specify '
                        'groups in .fit() function.')
            elif isinstance(X, DataFrame):
                if groups not in X.columns:
                    raise Exception(
                        'Default stratification column: ' +
                        str(groups) +
                        ' cannot be found in the origin data, please specify '
                        'groups in .fit() function.')


        split_index = self._process_split_start(split_start)
        graph_sections = cv_aux_info.graph_sections
        transforms = graph_sections.get('transform_nodes', [])
        pre_split_transforms = transforms[:split_index]
        post_split_transforms = transforms[split_index:]
        implicit_nodes = graph_sections['implicit_nodes']
        learner_node = graph_sections['learner_node'][0]

        # Pre-split section
        # data importer (if any) is always done pre-split
        steps = _PipelineSteps().add(graph_sections.get(
            'data_import')).add(pre_split_transforms)
        if len(steps.all_output_models) > 1:
            combine_model_node = transforms_modelcombiner(
                models=steps.all_output_models,
                output_model=cv_aux_info.output_model + '_combined_pre_split')
            steps.add(combine_model_node)

        # Post-split section
        cv_input_data = steps.last_output_data
        if cv_input_data is None:
            # Connect directly to input file, datastream, etc.
            # There should only be one input
            inputs = list(cv_aux_info.inputs.keys())
            assert len(inputs) == 1
            cv_input_data = '$' + inputs[0]

        cv_transform_models = steps.last_output_model
        cv_subgraph_input_data = '$cv_subgraph_input_data'  # Could be
        # any unique name
        cv_subgraph = _PipelineSteps().add(post_split_transforms).add(
            implicit_nodes)

        # Add learner node
        training_data = cv_subgraph.last_output_data
        learner_node.inputs['TrainingData'] = training_data
        learner_node.input_variables = {training_data}
        cv_subgraph.add(learner_node)

        if len(cv_subgraph.all_output_models) > 1:
            learner_model_new_name = cv_aux_info.output_model + '_learner'
            learner_node.outputs['PredictorModel'] = learner_model_new_name
            learner_node.output_variables = {learner_model_new_name}

            combine_model_node = transforms_manyheterogeneousmodelcombiner(
                transform_models=cv_subgraph.all_output_models[:-1],
                predictor_model=cv_subgraph.all_output_models[-1],
                model=cv_aux_info.predictor_model)
            cv_subgraph.add(combine_model_node)

        # Update the first data input of CV subgraph steps
        input_node, input_name = cv_subgraph.first_input_data
        input_node.inputs[input_name] = cv_subgraph_input_data
        input_node.input_variables = {cv_subgraph_input_data}

        cv_node = models_crossvalidator(
            data=cv_input_data,
            nodes=cv_subgraph,
            inputs_subgraph={'Data': cv_subgraph_input_data},
            outputs_subgraph={
                'PredictorModel': cv_aux_info.predictor_model},
            transform_model=cv_transform_models,
            kind=self._cv_kind,
            num_folds=cv,
            stratification_column=groups,
            predictor_model=self._get_output_name('predictor_model'),
            warnings=self._get_output_name('warnings'),
            overall_metrics=self._get_output_name('overall_metrics'),
            per_instance_metrics=self._get_output_name(
                'per_instance_metrics'),
            confusion_matrix=self._get_output_name('confusion_matrix'),
            label_column=label_column,
            weight_column=weights,
            group_column=group_id)

        steps.add(cv_node)
        graph = Graph(cv_aux_info.inputs, self.outputs, False, *steps)

        # prepare telemetry info
        class_name = type(self).__name__
        method_name = inspect.currentframe().f_code.co_name
        telemetry_info = ".".join([class_name, method_name])

        try:
            start_time = time.time()
            graph_run_results = graph.run(
                X=X,
                y=y,
                random_state=pipeline.random_state,
                seed=pipeline.random_state,
                w=weights,
                verbose=verbose,
                telemetry_info=telemetry_info,
                is_cv=True,
                output_types=self.output_types,
                **params)
        except RuntimeError as e:
            self._run_time = time.time() - start_time
            raise e

        self._raw_results = graph_run_results
        self._results = self._cleanup_results(graph_run_results, cv)
        return self._results
