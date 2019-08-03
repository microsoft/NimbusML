# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
import inspect
import itertools
import os
import tempfile
import time
import warnings
from collections import OrderedDict, namedtuple, defaultdict
from copy import deepcopy
from shutil import copyfile

import numpy as np
import six
from pandas import Categorical
from pandas import DataFrame, Series
from scipy.sparse import csr_matrix
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils.multiclass import unique_labels

from .internal.core.base_pipeline_item import BasePipelineItem
from .internal.entrypoints.data_customtextloader import \
    data_customtextloader
from .internal.entrypoints.models_anomalydetectionevaluator import \
    models_anomalydetectionevaluator
from .internal.entrypoints.models_binaryclassificationevaluator import \
    models_binaryclassificationevaluator
from .internal.entrypoints.models_classificationevaluator import \
    models_classificationevaluator
from .internal.entrypoints.models_clusterevaluator import \
    models_clusterevaluator
from .internal.entrypoints.models_datasettransformer import \
    models_datasettransformer
from .internal.entrypoints.models_rankingevaluator import \
    models_rankingevaluator
from .internal.entrypoints.models_regressionevaluator import \
    models_regressionevaluator
from .internal.entrypoints.models_summarizer import models_summarizer
from .internal.entrypoints.transforms_datasetscorer import \
    transforms_datasetscorer
from .internal.entrypoints.transforms_featurecombiner import \
    transforms_featurecombiner
from .internal.entrypoints.transforms_featurecontributioncalculationtransformer import \
    transforms_featurecontributioncalculationtransformer
from .internal.entrypoints.transforms_labelcolumnkeybooleanconverter \
    import \
    transforms_labelcolumnkeybooleanconverter
from .internal.entrypoints.transforms_labeltofloatconverter import \
    transforms_labeltofloatconverter
from .internal.entrypoints.transforms_manyheterogeneousmodelcombiner \
    import \
    transforms_manyheterogeneousmodelcombiner
from .internal.entrypoints.transforms_modelcombiner import \
    transforms_modelcombiner
from .internal.entrypoints.transforms_optionalcolumncreator import \
    transforms_optionalcolumncreator
from .internal.entrypoints \
    .transforms_predictedlabelcolumnoriginalvalueconverter import \
    transforms_predictedlabelcolumnoriginalvalueconverter
from .internal.entrypoints.transforms_scorecolumnselector import \
    transforms_scorecolumnselector
from .internal.entrypoints.transforms_texttokeyconverter import \
    transforms_texttokeyconverter
from .internal.utils.data_roles import Role, DataRoles
from .internal.utils.data_schema import DataSchema
from .internal.utils.data_stream import DataStream, ViewDataStream, \
    FileDataStream, BinaryDataStream
from .internal.utils.entrypoints import Graph
from .internal.utils.schema_helper import _extract_label_column
from .internal.utils.utils import trace, unlist


class TrainedWarning(UserWarning):
    """
    Raised when a trained model is trained again.
    """
    pass


class Pipeline:
    """

    Implementation of a pipeline.

    .. remarks::
        The Pipeline class assembles a pipeline of transforms, followed
        optionally by a trainer. The transforms need to
        implement fit() and transform() methods. The final trainer only
        needs to implement the fit() method.

        The Pipeline class only accepts trainers and transforms
        implemented in this package.

        The data sources for the methods may be a list, numpy.array,
        scipy.sparse_csr, pandas.DataFrame or a
        :py:func:`FileDataStream <nimbusml.FileDataStream>`.

        By default, the first transform will take all columns as input (
        i.e. will transform all columns), unless
        specific columns are requested
        (see `Columns </nimbusml/concepts/columns>`_ for
        how to specify columns to transform). The
        output column of the first transform is passed as the input
        column into the second transform for processing by
        default, unless the second transform requests a different column
        to operate on.

        The final trainer (if one exists) can select which columns to
        use for feature, labels, weights etc. See
        `Roles </nimbusml/concepts/roles#roles-and-learners>`_
        for more details on how to select these.

    :param steps: the list of operator or (name, operator) tuples  that
    are chained in the appropriate order.

    :param model: the path to the model file (".zip") if want to load a
    model directly from file (such as a trained model from ML.NET).

    :param random_state: the integer used as the random seed.

    .. seealso::
        :py:func:`FileDataStream <nimbusml.FileDataStream>`.
        :py:func:`DataSchema <nimbusml.DataSchema>`.
        :py:func:`Role <nimbusml.Role>`.

    """

    @trace
    def __init__(self, steps=None, model=None, random_state=None):
        if steps is not None:
            self._validate_steps(steps)
        self.steps = steps
        self.model = model
        self.random_state = random_state
        self._validate_schema()

    def clone(self):
        """
        Clones the pipeline and returns it in a non-trained state
        if the trained model was stored in a file on disk.
        You can clone the trained pipeline by running:
        ``Pipeline(**pipe.get_params())``.
        """
        cloned_steps = [deepcopy(s) for s in self.steps]

        # Rolls back role manipulation during fitting,
        # it removes attribute mapped to roles: label_column_name,
        # feature_column_name,
        # ...
        if len(cloned_steps) > 0:
            last_node = self.last_node
            if last_node.type != "transform":
                obj = cloned_steps[-1]
                if isinstance(obj, tuple):
                    obj = obj[-1]
                for role in DataRoles._allowed:
                    attr = Role.to_attribute(role)
                    attr_post_fit = attr + '_'
                    if hasattr(obj, attr_post_fit):
                        val_pre_fit = getattr(obj, attr_post_fit)
                        if role == Role.Feature:
                            if hasattr(obj, Role.Feature.lower()
                                       ) and obj.feature is not None:
                                val_pre_fit = deepcopy(obj.feature)
                            elif hasattr(obj, '_columns'):
                                if isinstance(obj._columns, list):
                                    val_pre_fit = obj._columns
                                elif isinstance(obj._columns, str):
                                    val_pre_fit = obj._columns
                                elif isinstance(obj._columns,
                                                dict) and role in \
                                        obj._columns:
                                    val_pre_fit = deepcopy(
                                        obj._columns[role])
                                else:
                                    val_pre_fit = None
                        setattr(obj, attr, val_pre_fit)
                        delattr(obj, attr_post_fit)
                for attr in ("input", "output"):
                    setattr(obj, attr, None)

        return Pipeline(
            steps=cloned_steps,
            model=None if isinstance(
                self.model,
                str) else self.model,
            random_state=self.random_state)

    def _clone_fitted(self):
        """
        Clones the pipeline in a trained state
        """
        cloned_steps = [deepcopy(s) for s in self.steps]
        return Pipeline(
            steps=cloned_steps,
            model=None if isinstance(
                self.model,
                str) else self.model,
            random_state=self.random_state)

    def get_params(self, deep=False):
        """
        Returns pipeline parameters

        :param deep: boolean, optional
            If True, will return the parameters for this pipeline and
            contained subobjects that are estimators.
        """
        out = dict(
            random_state=self.random_state,
            steps=self.steps,
            model=self.model)
        if not deep:
            return out
        named_estimators = filter(lambda x: isinstance(x, tuple),
                                  self.steps)
        out.update(named_estimators)
        for name, estimator in named_estimators:
            if estimator is None:
                continue
            for key, value in six.iteritems(
                    estimator.get_params(deep=True)):
                out['%s__%s' % (name, key)] = value
        return out

    def set_params(self, **params):
        """
        Set parameters to the pipeline.
        """
        if not params:
            return self

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition('__')
            if delim:
                nested_params[key][sub_key] = value
            else:
                # we come here if whole step is being replaced, for ex.
                # a new
                # transformer
                new_steps = []
                found = False
                for s in self.steps:
                    if isinstance(s, tuple) and s[0] == key:
                        new_steps.append((key, value))
                        found = True
                    else:
                        new_steps.append(s)
                if not found:
                    raise ValueError("step %s is not defined" % key)
                self.steps = new_steps

        valid_params = self.get_params(deep=True)
        for key, sub_params in nested_params.items():
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (key, self))
            valid_params[key].set_params(**sub_params)
        return self

    @property
    def nodes(self):
        nodes = []
        for s in self.steps:
            if isinstance(s, tuple):
                nodes.append(s[1])
            else:
                nodes.append(s)
        return nodes

    @property
    def last_node(self):
        if len(self.steps) <= 0:
            raise TypeError("No steps given.")
        last_step = self.steps[-1]
        return last_step if not isinstance(last_step, tuple) else \
            last_step[1]

    @trace
    def _validate_steps(self, steps):
        step_names = set()
        if len(steps) <= 0:
            raise TypeError("No steps given.")
        for i, s in enumerate(steps):
            if isinstance(s, tuple):
                # pipeline step
                obj = s[1]
                step_name = s[0]
                if step_name and not isinstance(step_name, str):
                    raise TypeError(
                        "All step names should be of string type. "
                        " '%s' (type %s) isn't" %
                        (step_name, type(step_name)))
                if step_name in step_names:
                    raise TypeError(
                        "All step names should be unique. '%s' isn't" %
                        (step_name))
                step_names.add(step_name)
            else:
                obj = s
            if not isinstance(obj, BasePipelineItem):
                raise TypeError(
                    "All steps should be of BasePipelineItem type. "
                    " '%s' (type %s) isn't" %
                    (obj, type(obj)))
            if not hasattr(obj, 'type'):
                raise AttributeError(
                    "Type is missing in node {0}".format(s))

    @trace
    def _validate_schema(self):
        """
        The function goes through every step and interprets
        every input can be infered from what the step receives.
        It deals with regular expressions, slices, dataframe...

        ::

            Transform1(COL('F[0-9]{1,2}'),  # produce various output
            following the pattern W[0-9]+
            Transform2(COL('F[0-9]{1,2}') + COL('W[0-9]+'),  # takes
            many columns
        """
        # Not yet implemented. Transform2 needs to receive the output of
        # the previous transform. The only issue with this mechanism
        # happens when ML.NET is needed to know the output column names
        # It usually produces only only columns (single or vector).
        pass

    @property
    def _IsSupervized(self):
        """Tells if the pipeline is supervisez or not.
        It returns None if the model was not trained, a boolean
        otherwise."""
        if self.steps is None:
            return None
        node_type = self.last_node.type
        return node_type != 'transform'

    def _preprocess_X_y(self, X, y=None, w=None):
        """
        Handles data preparation for fit, predict, transform
        """
        if isinstance(X, str):
            raise TypeError(
                "Filenames are not allowed, X must be a FileDataStream.")

        columns_renamed = False
        feature_columns = None
        label_column = None
        weight_column = None
        schema = None
        is_clone = False

        # Name of y is lost here.
        if isinstance(y, Series):
            label_column = y.name
        elif isinstance(y, DataFrame):
            if (y.columns.dtype == 'int64'):
                label_column = Role.Label
                y.columns = [label_column]
            else:
                label_column = y.columns[0]

        # check_array converts DataFrames and Series to ndarrays,
        # so capture column names
        # before calling it.
        if isinstance(X, DataFrame):
            if (X.columns.dtype == 'int64'):
                feature_columns = ['F' + str(x) for x in X.columns]
                X.columns = feature_columns
            else:
                feature_columns = list(X.columns)
            if y is not None and not isinstance(y, (str, tuple)):
                # Name of y is lost here.
                if isinstance(y, Series):
                    label_column = y.name
                elif isinstance(y, DataFrame):
                    if (y.columns.dtype == 'int64'):
                        label_column = Role.Label
                        y.columns = [label_column]
                    else:
                        label_column = y.columns[0]
        elif isinstance(X, Series):
            if X.name is None:
                feature_columns = ['F0']
                X.name = 'F0'
            else:
                feature_columns = [X.name]
            X = DataFrame(X)

        elif not isinstance(X, DataStream):
            if y is None or isinstance(y, (str, tuple)):
                X = check_array(
                    X,
                    accept_sparse=['csr'],
                    dtype=None,
                    ensure_2d=False,
                    force_all_finite=False)
            else:
                X, y = check_X_y(X, y, accept_sparse=['csr'],
                                 y_numeric=False, multi_output=True,
                                 dtype=None,
                                 ensure_2d=False, force_all_finite=False)

        # X --> Feature
        if isinstance(X, np.ndarray):
            X = DataFrame(X)
            if feature_columns is None:
                feature_columns = ['F' + str(x) for x in
                                   range(0, X.shape[1])]
                columns_renamed = True
            X.columns = feature_columns

        # y --> Label
        if isinstance(y, Categorical):
            if label_column is None:
                label_column = Role.Label
            y = DataFrame(data={label_column: y})
            columns_renamed = True

        elif isinstance(y, np.ndarray):
            y = DataFrame(y)
            if label_column is None:
                label_column = Role.Label
            y.columns = [label_column]
            columns_renamed = True

        elif isinstance(y, DataFrame):
            if label_column is None:
                if (y.columns.dtype == 'int64'):
                    label_column = Role.Label
                    y.columns = [label_column]
                else:
                    label_column = y.columns[0]

        elif isinstance(y, Series):
            y = DataFrame(y)
            if label_column is None:
                if (y.columns.dtype == 'int64'):
                    label_column = Role.Label
                    y.columns = [label_column]
                else:
                    label_column = y.columns[0]
            y.columns = [label_column]

        elif isinstance(y, list):
            y = DataFrame(y)
            if label_column is None:
                label_column = Role.Label
            y.columns = [label_column]

        elif isinstance(y, (str, tuple)):
            if isinstance(X, BinaryDataStream):
                if label_column is None:
                    label_column = y
            elif isinstance(X, DataStream):
                X = X if is_clone else X.clone()
                X._set_role(Role.Label, y)
                if label_column is None:
                    label_column = y
                y = None
            elif isinstance(X, DataFrame):
                name = y
                y = X[[y]]
                if label_column is None:
                    label_column = name
                X = X.drop(name, axis=1)
            else:
                raise NotImplementedError(
                    'If y is a column name, X must be a DataStream or a '
                    'DataFrame.')

        elif y is None and isinstance(X, DataStream) and X._has_role(
                Role.Label):
            label_column = X._get_role(Role.Label)

        elif y is not None:
            raise NotImplementedError('y type unsupported')

        # w --> Weight
        if w is not None:
            if isinstance(w, np.ndarray):
                w = DataFrame(w)
                if weight_column is None:
                    weight_column = Role.Weight
                w.columns = [weight_column]
                columns_renamed = True

            elif isinstance(w, Series):
                w = DataFrame(w)
                if weight_column is None:
                    weight_column = w.columns[0]
                w.columns = [weight_column]

            elif isinstance(w, (str, tuple)):
                if isinstance(X, DataStream):
                    X = X if is_clone else X.clone()
                    X._set_role(Role.Weight, w)
                    weight_column = w
                    w = None
                elif isinstance(w, DataFrame):
                    name = w
                    w = X[[w]]
                    if weight_column is None:
                        weight_column = name
                    X = X.drop(name, axis=1)
                else:
                    raise NotImplementedError(
                        'If w is a column name, X must be a DataStream.')

            elif w is None and isinstance(X, DataStream) and X._has_role(
                    Role.Weight):
                if weight_column is None:
                    weight_column = X._get_role(Role.Weight)

        # construct schema if necessary (DataFrame, Array, Categorical)
        if columns_renamed:
            schema = DataSchema.read_schema(X, y, w)
        elif schema is None:
            if isinstance(X, DataStream):
                schema = X.schema
            elif isinstance(X, ViewDataStream):
                schema = X.parent.schema
            elif isinstance(X, (DataFrame, csr_matrix)):
                schema = DataSchema.read_schema(X, y, w)
            else:
                raise ValueError(
                    "schema cannot be determined for X ({0}) and y ({"
                    "1}).".format(type(X), type(y)))

        return X, y, columns_renamed, feature_columns, label_column, \
            schema, w, weight_column

    def _init_graph_nodes(
            self,
            X,
            y,
            input_data,
            file_data,
            schema,
            feature_columns,
            label_column,
            output_data,
            output_model,
            strategy_iosklearn):
        inputs = OrderedDict([(input_data.replace('$', ''), '')])
        graph_nodes = OrderedDict()
        if isinstance(X, FileDataStream):
            import_text_node = data_customtextloader(
                input_file=file_data,
                custom_schema=schema.to_string(add_sep=True),
                data=input_data)
            import_text_node._implicit = True
            graph_nodes['data_import'] = [import_text_node]
            inputs = OrderedDict([(file_data.replace('$', ''), '')])

        # connect transform node inputs/outputs
        if feature_columns is None and not isinstance(X, BinaryDataStream):
            if schema is None:
                schema = DataSchema.read_schema(X)
            feature_columns = [c.Name for c in schema]
            if label_column:
                # if label_column is a string, remove it from
                # feature_columns
                if isinstance(label_column, (str, six.text_type)):
                    if label_column in feature_columns:
                        feature_columns.remove(label_column)
                # if label_column is not a string (list then), remove
                # all from
                # feature_columns
                else:
                    for col in label_column:
                        if col in feature_columns:
                            feature_columns.remove(col)
            elif y is not None:
                if isinstance(y, DataFrame):
                    feature_columns.remove(y.columns)
                elif isinstance(y, Series):
                    feature_columns.remove(y.name)
                else:
                    raise ValueError(
                        "y parameter can be only a DataFrame or a Series "
                        "or a string not {0}.".format(
                            type(y)))

        (transform_nodes, columns_out) = self._process_transformers(
            input_data=input_data,
            input_columns=feature_columns,
            label_column=label_column,
            output_data=output_data,
            output_model=output_model,
            strategy_iosklearn=strategy_iosklearn)
        graph_nodes['transform_nodes'] = transform_nodes
        return graph_nodes, feature_columns, inputs, transform_nodes, \
            columns_out

    def _update_graph_nodes_for_learner(
            self,
            graph_nodes,
            transform_nodes,
            columns_out,
            label_column,
            weight_column,
            output_data,
            output_model,
            predictor_model,
            y,
            strategy_iosklearn):
        last_node = self.last_node  # could be predictor or transformer
        if last_node.type != 'transform':  # last node is predictor
            if hasattr(
                    last_node,
                    'feature_column_name') and last_node.feature_column_name is \
                    not None:
                if isinstance(last_node.feature_column_name, list):
                    learner_features = last_node.feature_column_name
                    last_node.feature_column_name = 'Features'
                else:
                    learner_features = [last_node.feature_column_name]
            elif strategy_iosklearn in ("previous", "accumulate"):
                if hasattr(
                        last_node,
                        'feature') and last_node.feature is not None:
                    if isinstance(last_node.feature, list):
                        learner_features = last_node.feature
                    else:
                        learner_features = [last_node.feature]
                    last_node.feature_column_name = 'Features'
                elif isinstance(columns_out, list):
                    learner_features = columns_out
                    last_node.feature_column_name = 'Features'
                elif columns_out is None:
                    learner_features = ['Features']
                    last_node.feature_column_name = 'Features'
                else:
                    learner_features = [columns_out]
                    last_node.feature_column_name = 'Features'
            else:
                raise NotImplementedError(
                    "Strategy '{0}' to handle unspecified inputs is not "
                    "implemented".format(
                        strategy_iosklearn))

            if label_column is not None or last_node._use_role(Role.Label):
                if getattr(last_node, 'label_column_name_', None):
                    label_column = last_node.label_column_name_
                elif getattr(last_node, 'label_column_name', None):
                    label_column = last_node.label_column_name
                elif label_column:
                    last_node.label_column_name = label_column
                elif y is None:
                    if label_column is None:
                        label_column = Role.Label
                    last_node.label_column_name = label_column
                else:
                    label_column = _extract_label_column(
                        last_node, DataSchema.read_schema(y))
                    if label_column is None:
                        label_column = Role.Label
                    last_node.label_column_name = label_column
            else:
                last_node.label_column_name = None
                label_column = None

            if weight_column is not None or last_node._use_role(
                    Role.Weight):
                if getattr(last_node, 'example_weight_column_name', None):
                    weight_column = last_node.example_weight_column_name
                elif weight_column:
                    last_node.example_weight_column_name = weight_column
            else:
                last_node.example_weight_column_name = None
                weight_column = None

            if (hasattr(last_node, 'row_group_column_name_')
                    and last_node.row_group_column_name_ is not None):
                group_id_column = last_node.row_group_column_name_
            elif (hasattr(last_node,
                          'row_group_column_name') and
                  last_node.row_group_column_name is not None):
                group_id_column = last_node.row_group_column_name
            else:
                group_id_column = None

            # Training.
            implicit_nodes = self._process_learner(
                learner=last_node,
                features=learner_features,
                label=label_column,
                weight=weight_column,
                num_transforms=len(transform_nodes),
                output_data=output_data,
                output_model=output_model)
            graph_nodes['implicit_nodes'] = implicit_nodes

            # Check roles
            last_node._check_roles()

            # todo: ideally all the nodes have the same name for params
            # so we dont have to distinguish if its learner or
            # transformer. We will supply
            # input_data, output_data & output_model vars. Its up to
            # node to
            # use suplied vars
            learner_node = last_node._get_node(
                feature_column_name=learner_features,
                training_data=output_data,
                predictor_model=predictor_model,
                label_column_name=label_column,
                example_weight_column_name=weight_column,
                row_group_column_name=group_id_column)
            graph_nodes['learner_node'] = [learner_node]
            return graph_nodes, learner_node, learner_features
        else:
            return graph_nodes, None, None

    def _fit_graph(self, X, y, verbose, **params):
        # start the clock!
        start_time = time.time()
        self.verbose = verbose

        params.pop('graph_id', None)
        max_slots = params.pop('max_slots', -1)
        weights = params.pop('weight', None)
        strategy_iosklearn = params.pop('iosklearn', 'previous')
        do_fit_transform = params.pop('do_fit_transform', False)
        params.pop('output_scores', False)
        output_binary_data_stream = params.pop(
            'output_binary_data_stream', False)
        params.pop('parallel', None)

        X, y, columns_renamed, feature_columns, label_column, schema, \
            weights, weight_column = self._preprocess_X_y(X, y, weights)

        self._check_ambiguities(X, y, weights)

        # A variable is a string beginning with a $, and it signifies
        # that a specific
        # input or output of a node needs to be loaded or saved during the
        # graph execution.
        file_data = "$file"
        input_data = "$input_data"
        output_data = "$output_data"
        output_model = "$output_model"
        predictor_model = "$predictor_model"

        graph_nodes, feature_columns, inputs, transform_nodes, \
            columns_out = \
            self._init_graph_nodes(
                X, y, input_data, file_data, schema,
                feature_columns, label_column, output_data, output_model,
                strategy_iosklearn=strategy_iosklearn)

        # see if the is a learner at the end
        graph_nodes, learner_node, learner_features = \
            self._update_graph_nodes_for_learner(
                graph_nodes,
                transform_nodes,
                columns_out, label_column,
                weight_column,
                output_data, output_model,
                predictor_model, y,
                strategy_iosklearn=strategy_iosklearn)

        # graph_nodes contain graph sections, which is needed for CV.
        # Save it, then flatten it, which is what the rest of the code
        # expects.
        graph_sections = graph_nodes
        graph_nodes = list(itertools.chain(*graph_nodes.values()))

        # combine output models
        transform_models = [node.outputs["Model"]
                            for node in graph_nodes if
                            "Model" in node.outputs]
        if learner_node and len(
                transform_models) > 0:  # no need to combine if there is
            #  only 1 model
            combine_model_node = transforms_manyheterogeneousmodelcombiner(
                transform_models=transform_models,
                predictor_model=(
                    predictor_model if learner_node else None),
                model=output_model)
            combine_model_node._implicit = True
            graph_nodes.append(combine_model_node)
        elif len(transform_models) > 1:
            combine_model_node = transforms_modelcombiner(
                models=transform_models,
                output_model=output_model)
            combine_model_node._implicit = True
            graph_nodes.append(combine_model_node)
        elif len(graph_nodes) == 0:
            raise RuntimeError(
                "Unable to process the pipeline len(transform_models)={"
                "0}.".format(
                    len(transform_models)))

        # create the graph
        outputs = OrderedDict([(output_model.replace('$', ''), '')])
        # REVIEW: ideally we should remove output completely from the
        # graph if its not needed
        # however graph validation logic prevents doing that at the moment,
        # revisit this at later point, bug# 249112
        if learner_node is None:  # last node is transformer
            outputs[output_data.replace(
                '$', '')] = '' if do_fit_transform else '<null>'
        graph = Graph(
            inputs,
            outputs,
            do_fit_transform and output_binary_data_stream,
            *(graph_nodes))

        # Checks that every parameter in params was used.
        if len(params) > 0:
            raise ValueError(
                "Following parameters were not used: {0}".format(params))

        # prepare telemetry info
        class_name = type(self).__name__
        method_name = inspect.currentframe().f_code.co_name
        telemetry_info = ".".join([class_name, method_name])

        CvAuxilaryInfo = namedtuple('CvAuxilaryInfo',
                                    ['graph_sections',
                                     'inputs',
                                     'output_data',
                                     'output_model',
                                     'predictor_model',
                                     'label_column'])
        cv_aux_info = CvAuxilaryInfo(
            graph_sections=graph_sections,
            inputs=inputs,
            output_data=output_data,
            output_model=output_model,
            predictor_model=predictor_model,
            label_column=label_column)
        return graph, X, y, weights, start_time, schema, telemetry_info, \
            learner_features, cv_aux_info, max_slots

    def get_fit_info(self, X, y=None, **params):
        """
        Returns information about the pipeline.

        :param X: {array-like [n_samples, n_features],
            :py:func:`FileDataStream <nimbusml.FileDataStream>` }
        :param y: {array-like [n_samples]}
        :return: tuple (list of dictonaries, list of entrypoints),
            both lists do not necessarily have the same length

        .. remarks::
            The first list is the list of operators the user defines.
            In that case, the result is a list of dictionaries with keys
            *operator*, *name*,
            *inputs*, *outputs*, *type*, *current_schema*. The last is
            the schema
            after the transform or a learner is applied.

            The second list is what *nimbusml* internally uses. The number
            of entrypoint may be different from the list of operators. This
            information is mostly used by contributors.

        Example: ``pipe.get_fit_info(X,Y)``.

        """

        def process_input_output(classname, node, input_schema):

            if 'NGramFeaturizer' in classname:
                suffix = ["", "_TransformedText"]
            else:
                suffix = [""]

            inp = node.inputs
            inputs = []
            outputs = []
            if 'Column' in inp:
                cols = inp['Column']
                if isinstance(cols, dict):
                    # Transform column Source into Name.
                    if 'Source' in cols and 'Name' in cols:
                        if isinstance(cols["Source"], list):
                            inputs.extend(cols["Source"])
                        else:
                            inputs.append(cols["Source"])
                        if isinstance(cols["Name"], list):
                            raise TypeError(
                                'Output cannot be a list with this type '
                                'of schema.\n{0}'.format(
                                    node))
                        for s in suffix:
                            outputs.append(cols["Name"] + s)
                    else:
                        raise NotImplementedError(
                            "Not implemented for cols={0}\n{1}".format(
                                cols, node))
                else:
                    # Multiple columns to transform.
                    for ei in cols:
                        if isinstance(ei, dict):
                            if 'Name' in ei and 'Source' in ei:
                                if isinstance(ei["Source"], list):
                                    inputs.extend(ei['Source'])
                                else:
                                    inputs.append(ei['Source'])
                                if isinstance(ei['Name'], list):
                                    for s in suffix:
                                        for o in ei['Name']:
                                            outputs.append(o + s)
                                else:
                                    for s in suffix:
                                        outputs.append(ei['Name'] + s)
                            else:
                                raise NotImplementedError(str(node))
                        elif isinstance(ei, str):
                            # Input = Output.
                            inputs.append(ei)
                            for s in suffix:
                                outputs.append(ei + s)
                        else:
                            raise NotImplementedError(
                                "Not implemented for ei={0}\n{1}".format(
                                    ei, node))
            else:
                assigned = []
                for role in sorted(DataRoles._allowed):
                    attr = DataRoles.to_parameter(role)
                    if attr in inp:
                        assigned.append(inp[attr])
                assigned = set(assigned)
                not_assigned = [
                    col for col in input_schema if col not in assigned]

                for role in sorted(DataRoles._allowed):
                    attr = DataRoles.to_parameter(role)
                    if attr in inp:
                        if attr == 'FeatureColumnName' and inp[attr]\
                                not in input_schema:
                            val = not_assigned
                        else:
                            val = inp[attr]
                        if isinstance(val, (str, tuple)):
                            inputs.append('{0}:{1}'.format(role, val))
                        else:
                            inputs.append(
                                '{0}:{1}'.format(
                                    role, ",".join(val)))

            return inputs, outputs

        if y is None:
            new_pipe = self.clone()
        else:
            new_pipe = self._clone_fitted()
        new_nodes = new_pipe._nodes_with_presteps(new_pipe.nodes)
        graph, X, y, weights, _, schema, telemetry_info, \
            learner_features, _, _ = \
            new_pipe._fit_graph(X, y, verbose=0, **params)

        # entrypoints
        entrypoints = [_ for _ in graph]

        # operators
        schema = [_.Name for _ in schema] if schema is not None else []
        sch = list(schema)
        info = [
            dict(
                name=None,
                schema_after=sch,
                type='start',
                operator=None,
                outputs=sch)]
        explicit = [
            node for node in entrypoints if not hasattr(
                node, "_implicit")]
        if len(explicit) != len(new_nodes):
            raise RuntimeError(
                "{0} != {1}: unexpected number of explicit steps".format(
                    len(explicit), len(new_nodes)))
        nodes = [(n, node) for n, node in zip(new_nodes, explicit)]
        current_schema = schema
        for node, entrypoint in nodes:
            if 'ColumnDropper' in node.__class__.__name__:
                schi = list(current_schema)
                for co in entrypoint.inputs['DropColumns']:
                    if co in current_schema:
                        del current_schema[current_schema.index(co)]
                    else:
                        raise ValueError(
                            "Unable to find column '{0}' in schema {"
                            "1}".format(
                                co, current_schema))
                sch = list(current_schema)
                info.append(
                    dict(
                        operator=node,
                        name=node.__class__.__name__,
                        type=node.type,
                        schema_after=sch,
                        inputs=schi,
                        outputs=sch))
            else:
                inp, out = process_input_output(
                    node.__class__.__name__, entrypoint, current_schema)
                if node.type == 'transform':
                    for o in out:
                        if o not in current_schema:
                            current_schema.append(o)
                else:
                    if learner_features:
                        inp0 = inp
                        inp = []
                        for c in inp0:
                            if c.startswith('Feature:'):
                                inp.append(
                                    'Feature:' +
                                    ','.join(
                                        sorted(learner_features)))
                            else:
                                inp.append(c)
                    if node.type in ('regressor', 'ranker', 'anomaly'):
                        current_schema = ['Score']
                    elif node.type == 'classifier':
                        current_schema = [
                            'PredictedLabel', 'PredictedProba', 'Score']
                    elif node.type == 'clusterer':
                        if hasattr(node, 'n_clusters'):
                            current_schema = [
                                                 'PredictedLabel'] + [
                                                 'Score.%d' % i for i in
                                                 range(node.n_clusters)]
                        else:
                            raise AttributeError(
                                "Unable to guess the number of predicted "
                                "clusters for node {0}".format(
                                    node))
                    else:
                        raise NotImplementedError(
                            "Unable to give output schema for type='{"
                            "0}'".format(
                                node.type))
                    out = list(current_schema)

                info.append(
                    dict(
                        operator=node,
                        name=node.__class__.__name__,
                        inputs=inp,
                        type=node.type,
                        outputs=out,
                        schema_after=list(current_schema)))
        operators = info

        # return
        return (operators, entrypoints)

    def _get_last_node_roles(self, X, y=None, **params):
        inputs = self.get_fit_info(X, y, **params)[0][-1]['inputs']
        # in case ":" in column names
        return dict(map(lambda s: s.split(':', 1), inputs))

    @trace
    def fit(self, X, y=None, verbose=1, **params):
        """
        Fit the pipeline.

        :param X: {array-like [n_samples, n_features],
           :py:func:`FileDataStream <nimbusml.FileDataStream>` }
        :param y: {array-like [n_samples]}

        Example:
           .. literalinclude::
             /../nimbusml/examples/Pipeline.py
                  :language: python
           .. literalinclude::
             /../nimbusml/examples/PipelineWithGridSearchCV1.py
                  :language: python
           .. literalinclude::
             /../nimbusml/examples/PipelineWithGridSearchCV2.py
                  :language: python

        """
        if self._is_fitted:
            # We restore the initial steps as they were
            # modified by the previous training.
            if y is not None:
                clone = self._clone_fitted()
            else:
                clone = self.clone()
            self.steps = clone.steps

        # Clear cached values
        for attr in ["_run_time_error", "model_summary"]:
            if hasattr(self, attr):
                delattr(self, attr)

        # Caches the predictor to restore it as it was
        # in case of exception. It is deleted after the training.
        self._cache_predictor = deepcopy(self.steps[-1])

        # Checks that no node was ever trained.
        for i, n in enumerate(self.nodes):
            if hasattr(n, "model_") and n.model_ is not None:
                warnings.warn(
                    'Step {0}: {1} was already trained. Its coefficients '
                    'will be overwritten. Use clone() to get an '
                    'untrained version of it.'.format(
                        i, n.__class__.__name__), TrainedWarning)
                break

        self._extract_classes(y)

        graph, X, y, weights, start_time, schema, telemetry_info, \
            learner_features, _, max_slots = self._fit_graph(
                X, y, verbose, **params)
        params.pop('max_slots', max_slots)

        def move_information_about_roles_once_used():
            last_node = self.last_node
            for role in DataRoles._allowed:
                name = Role.to_attribute(role)
                for obj in [self, last_node]:
                    if hasattr(obj, name):
                        name2 = Role.to_attribute(role)
                        setattr(obj, name2 + "_", getattr(obj, name))
                        del obj.__dict__[name]

        # run the graph
        # REVIEW: we should have the possibility to keep the model in
        # memory
        # and not in a file.
        try:
            (out_model, out_data, out_metrics) = graph.run(
                X=X,
                y=y,
                random_state=self.random_state,
                w=weights,
                verbose=verbose,
                max_slots=max_slots,
                telemetry_info=telemetry_info,
                **params)
        except RuntimeError as e:
            self._run_time = time.time() - start_time
            if hasattr(e, 'model'):
                self.model = e.model
            # We restore the initial steps as they were modified
            # by the fitting function.
            move_information_about_roles_once_used()
            clone = self.clone()
            self.steps = clone.steps
            self._run_time_error = e
            self.steps[-1] = self._cache_predictor
            delattr(self, "_cache_predictor")
            raise e

        move_information_about_roles_once_used()
        self.graph_ = graph
        self.model = out_model
        self.data = out_data
        # stop the clock
        self._run_time = time.time() - start_time
        self._write_csv_time = graph._write_csv_time
        delattr(self, "_cache_predictor")
        return self

    @trace
    def fit_transform(
            self,
            X,
            y=None,
            verbose=0,
            as_binary_data_stream=False,
            **params):
        """
        If a pipeline only has transforms, returns transformed data as a
        pandas dataframe

        :param X: {array-like [n_samples, n_features],
            :py:func:`FileDataStream <nimbusml.FileDataStream>` }
        :param y: {array-like [n_samples]}
        """
        self.fit(
            X,
            y,
            verbose,
            do_fit_transform=True,
            output_binary_data_stream=as_binary_data_stream,
            **params)
        return self.data

    def _nodes_with_presteps(self, nodes):
        """
        Some nodes implement method *_nodes_with_presteps*
        to append preprocessing node before this one.
        One particular case: C# is strict about types, Python is
        less strict. This method can be used to insert a *TypeConverter*
        before a *MinMaxScaler*.
        """
        res = []
        for n in nodes:
            if isinstance(n, tuple):
                pre = n[-1]._nodes_with_presteps()
                for i, p in enumerate(pre):
                    if isinstance(p, tuple):
                        res.append(p)
                    else:
                        res.append(("%spre%d" % (n[0], i), p))
            else:
                pre = n._nodes_with_presteps()
                res.extend(pre)
        return res

    @trace
    def _process_transformers(self, input_data, input_columns, output_data,
                              output_model, label_column,
                              strategy_iosklearn):
        """
        Connect transform nodes.
        """
        last_node = self.last_node  # could be predictor or transformer
        if isinstance(last_node, tuple):
            last_node = last_node[1]
        all_transformers = False
        if last_node.type == 'transform':
            transformers = self._nodes_with_presteps(self.nodes)
            all_transformers = True
        else:
            transformers = self._nodes_with_presteps(self.nodes[:-1])

        nodes = []
        num_transforms = len(transformers)
        columns_out_prev = input_columns
        columns_out = input_columns
        columns_in = None
        for i in range(num_transforms):
            data_in = input_data if i == 0 else output_data + str(i)
            data_out = output_data if all_transformers and i == (
                    num_transforms - 1) else output_data + str(i + 1)
            model_out = output_model if all_transformers and \
                num_transforms == 1 else output_model + str(i + 1)

            # set input/output
            # if no input set on a node, then take output from previous
            # node
            if hasattr(
                    transformers[i],
                    'input') and transformers[i].input is not None:
                columns_in = transformers[i].input
            else:
                columns_in = columns_out_prev
                if isinstance(columns_in, (str, tuple)):
                    columns_in = [columns_in]
                elif transformers[i].__class__.__name__ \
                        in ['ColumnConcatenator']:
                    columns_in = [columns_in]

            # if no output set on a node, then assume it outputs the
            # same set of columns as it takes in
            if isinstance(columns_in, (ViewDataStream, DataStream)):
                raise RuntimeError(
                    'input_columns must be converted in a list of '
                    'string, it is still a DataStream.')
            if hasattr(
                    transformers[i],
                    'output') and transformers[i].output is not None:
                if strategy_iosklearn == 'previous':
                    columns_out = transformers[i].output
                elif strategy_iosklearn == 'accumulate':
                    columns_out = list(
                        sorted(set(columns_out + transformers[i].output)))
                else:
                    raise NotImplementedError(
                        "Strategy '{0}' to handle unspecified inputs is "
                        "not implemented".format(
                            strategy_iosklearn))
            elif 'ColumnDropper' == transformers[i].__class__.__name__:
                columns_out = [c for c in columns_out if
                               c not in columns_in]
            else:
                if transformers[i].__class__.__name__ in \
                        ['NGramFeaturizer']:
                    cout = columns_in[0]
                else:
                    cout = columns_in
                if strategy_iosklearn == 'previous':
                    columns_out = cout
                elif strategy_iosklearn == 'accumulate':
                    columns_out = list(
                        sorted(set(cout + columns_out_prev)))
                else:
                    raise NotImplementedError(
                        "Strategy '{0}' to handle unspecified inputs is "
                        "not implemented".format(
                            strategy_iosklearn))

            step = transformers[i]
            if isinstance(step, tuple):
                step = step[1]
            node = step._get_node(data=data_in, input=columns_in,
                                  output_data=data_out,
                                  output=columns_out, model=model_out,
                                  label_column_name=label_column)
            if isinstance(node, list):
                # In most cases, _get_node returns only one entrypoint
                # mapped to the current step. In rare cases, the python
                # library enables one usage not implemented in
                # Microsoft.ML. That is the case for entrypoint PCA
                # which assumes its inputs are one vector column and not
                #  multiple columns. However, in that case, The python
                # package assumes the user wants to compute a PCA on the
                #  concatenation of all provided columns.
                # This can be done by adding a ColumnConcatenator just
                # before the PCA.
                # In that case, the method _get_node returns two nodes:
                # [ColumnConcatenator, PCA] and not one. The first one
                # is marked as implicit as it implicitely enables one
                # non-ambiguous usage.
                # Other usage could be implemented the same way but the
                # list should be kept short as it usually makes it more
                # difficult to investigate an issue.
                nodes.extend(node)
                inputs = node[0].inputs
                input_variables = {
                    x for x in unlist(inputs.values())
                    if isinstance(x, str) and x.startswith("$")}
                outputs = node[-1].outputs
                output_variables = {
                    x for x in unlist(outputs.values())
                    if isinstance(x, str) and x.startswith("$")}
                node[0].input_variables = input_variables
                node[-1].output_variables = output_variables
                nb_explicit = sum(
                    [0 if hasattr(n, "_implicit") else 1 for n in node])
                if nb_explicit != 1:
                    raise RuntimeError(
                        "Every node must be implicit except one which "
                        "corresponds to entrypoint '{0}'".format(
                            type(step)))
            else:
                nodes.append(node)
                inputs = node.inputs
                input_variables = {
                    x for x in unlist(inputs.values())
                    if isinstance(x, str) and x.startswith("$")}
                outputs = node.outputs
                output_variables = {
                    x for x in unlist(outputs.values())
                    if isinstance(x, str) and x.startswith("$")}
                node.input_variables = input_variables
                node.output_variables = output_variables

            columns_out_prev = columns_out

        return (nodes, columns_out)

    @trace
    def _process_learner(
            self,
            learner,
            features,
            label,
            num_transforms,
            output_data,
            output_model,
            weight=None):
        if learner.type == 'regressor':
            optional_node = transforms_optionalcolumncreator(
                column=[label],
                data="$input_data" if num_transforms == 0 else
                output_data +
                str(
                    num_transforms),
                output_data="$optional_data",
                model=output_model + str(num_transforms + 1))
            optional_node._implicit = True
            label_node = transforms_labeltofloatconverter(
                data="$optional_data",
                label_column=label,
                output_data="$label_data",
                model=output_model + str(
                    num_transforms + 2))
            label_node._implicit = True
            feature_node = transforms_featurecombiner(
                data="$label_data",
                features=features,
                output_data=output_data,
                model=output_model + str(
                    num_transforms + 3))
            feature_node._implicit = True
            implicit_nodes = [optional_node, label_node, feature_node]
        elif learner.type in ('classifier', 'ranker'):
            optional_node = transforms_optionalcolumncreator(
                column=[label],
                data="$input_data" if num_transforms == 0 else
                output_data +
                str(
                    num_transforms),
                output_data="$optional_data",
                model=output_model + str(num_transforms + 1))
            optional_node._implicit = True
            label_node = transforms_labelcolumnkeybooleanconverter(
                data="$optional_data",
                label_column=label,
                output_data="$label_data",
                text_key_values=False,
                model=output_model + str(num_transforms + 2))
            label_node._implicit = True

            feature_node = transforms_featurecombiner(
                data="$label_data",
                features=features,
                output_data=output_data,
                model=output_model + str(
                    num_transforms + 3))
            feature_node._implicit = True
            implicit_nodes = [optional_node, label_node, feature_node]
        elif learner.type in {'recommender', 'sequence'}:
            raise NotImplementedError(
                "Type '{0}' is not implemented yet.".format(
                    learner.type))
        else:
            feature_node = transforms_featurecombiner(
                data="$input_data" if num_transforms == 0 else
                output_data +
                str(
                    num_transforms),
                features=features,
                output_data=output_data,
                model=output_model + str(num_transforms + 1))
            feature_node._implicit = True
            implicit_nodes = [feature_node]

        return implicit_nodes

    @trace
    def _fix_ranking_metrics_schema(self, out_metrics):
        out_metrics.columns = ['NDCG@1', 'NDCG@2', 'NDCG@3',
                               'DCG@1', 'DCG@2', 'DCG@3', ]
        return out_metrics

    def _evaluation_infer(self, evaltype, label_column, group_id,
                          **params):
        all_nodes = []
        if not self.steps:
            if evaltype == 'auto':
                raise ValueError(
                    "need to specify 'evaltype' explicitly if model is "
                    "loaded")
        common_eval_args = OrderedDict(data="$scoredVectorData",
                                       overall_metrics="$output_metrics",
                                       score_column="Score",
                                       label_column=label_column)
        params.update(common_eval_args)

        type_ = self._last_node_type() if evaltype == 'auto' else evaltype

        if type_ == 'binary':
            all_nodes.extend(
                [models_binaryclassificationevaluator(**params)])

        elif type_ == 'multiclass':
            all_nodes.extend(
                [models_classificationevaluator(**params)])

        elif type_ in ['regressor', 'regression']:
            all_nodes.extend([models_regressionevaluator(**params)])

        elif type_ in ['clusterer', 'cluster']:
            label_node = transforms_labelcolumnkeybooleanconverter(
                data="$scoredVectorData", label_column=label_column,
                output_data="$label_data")
            clustering_eval_args = OrderedDict(
                data="$label_data",
                overall_metrics="$output_metrics",
                score_column="Score",
                label_column=label_column)
            params.update(clustering_eval_args)
            all_nodes.extend([label_node,
                              models_clusterevaluator(**params)
                              ])

        elif type_ == 'anomaly':
            label_node = transforms_labelcolumnkeybooleanconverter(
                data="$scoredVectorData", label_column=label_column,
                output_data="$label_data")
            anom_eval_args = OrderedDict(
                data="$label_data",
                overall_metrics="$output_metrics",
                score_column="Score",
                label_column=label_column
            )
            params.update(anom_eval_args)
            all_nodes.extend(
                [label_node,
                 models_anomalydetectionevaluator(**params)])

        elif type_ == 'ranking':
            svd = "$scoredVectorData"
            column = [OrderedDict(Source=group_id, Name=group_id)]
            algo_args = dict(data=svd, output_data=svd, column=column)
            key_node = transforms_texttokeyconverter(**algo_args)
            evaluate_node = models_rankingevaluator(
                group_id_column=group_id, **params)
            all_nodes.extend([
                key_node,
                evaluate_node
            ])

        else:
            raise ValueError(
                "%s is not a valid type for evaluation." %
                evaltype)

        return all_nodes

    def _last_node_type(self):
        last_node = self.last_node

        if last_node.type != 'classifier':
            # For everything other than classifier, type can be used as-is
            return last_node.type

        # For classifier, we need to inspect the class name
        last_node_class_name = type(last_node).__name__

        if 'Binary' in last_node_class_name:
            return 'binary'
        else:
            return 'multiclass'

    @property
    def _is_fitted(self):
        """
        Tells if the pipeline was trained.
        """
        if not hasattr(self, 'model'):
            return False
        if self.model is None or not os.path.isfile(self.model):
            return False
        if hasattr(self, "_run_time_error"):
            return False
        return True

    def __len__(self):
        """
        Returns the pipeline length.
        """
        if not hasattr(self, 'steps') or self.steps is None:
            return 0
        return len(self.steps)

    def __delitem__(self, index):
        """
        Removes one element of the pipeline.
        *index* must be an integer or a stepname.
        In that case, the method will look for all transforms
        or learners for which the stepname is equal to *index*.
        """
        if self._is_fitted:
            raise RuntimeError(
                "Model is fitted and cannot be modified. You should "
                "clone it and then modify.")
        if len(self) == 0:
            raise IndexError("Pipeline is empty.")
        if isinstance(index, six.integer_types):
            del self.steps[index]
        elif isinstance(index, str):
            res = []
            for i, n in enumerate(self.steps):
                if isinstance(n, tuple) and n[0] == index:
                    res.append(i)
            for i in reversed(res):
                del self.steps[i]
        else:
            raise TypeError(
                "index must be an integer or a stepname as a string")

    def append(self, step):
        """
        Extends the pipeline with a new transform/learner at the end.
        Note that a fitted pipeline cannot be modified.
        Example: ``pipe.append(FastLinearRegressor())``.

        Example: ``pipe.append(("learner", FastLinearRegressor()))``.

        :param step: the transform/learner to append
        """
        if self._is_fitted:
            raise RuntimeError(
                "Model is fitted and cannot be modified. You should "
                "clone it and then modify.")
        if self.steps is None:
            self.steps = []
        self.steps.append(step)

    def insert(self, pos, step):  # todo sweep
        """
        Inserts a transform/learner into the pipeline.

        :param pos: position to insert, should be integers
        :param step: the transform/learner to insert

        Example: ``pipe.insert(1, FastLinearRegressor())``.

        Example: ``pipe.insert(1, ("learner", FastLinearRegressor()))``.
        """
        if self._is_fitted:
            raise RuntimeError(
                "Model is fitted and cannot be modified. You should "
                "clone it and then modify.")
        if self.steps is None:
            self.steps = []
        self.steps.insert(pos, step)

    def __getitem__(self, index):
        """
        Returns one element of the pipeline.
        *index* must be an integer or a stepname.
        In that case, the method will look for all transforms
        or learners for which the stepname is equal to *index*.
        The returned objets contains information before training happened.
        Output from a specific step cannot be computed from the returned
        object.

        .. index:: entrypoint

        By extension, this method returns a description object
        which describs what the training should do.
        Once trained, the attriutes ``graph_.nodes`` returned
        the execution objects (or *entrypoints*). Their number
        can be different, they represent how a pipeline the user defines
        is translated into core transforms and learners.
        """
        if len(self) == 0:
            raise IndexError("Pipeline is empty.")
        if isinstance(index, six.integer_types):
            return self.steps[index]
        elif isinstance(index, str):
            res = []
            for n in self.steps:
                if isinstance(n, tuple) and n[0] == index:
                    res.append(n)
            return res
        else:
            raise TypeError(
                "index must be an integer or a stepname as a string")

    def _check_ambiguities(self, X, y_temp, weights):
        if y_temp is not None:
            def getn(n):
                if isinstance(n, tuple):
                    return n[-1]
                else:
                    return n

            has_defined = any(
                n.has_defined_columns(
                    Role.Label) for n in map(
                    getn,
                    self.nodes))
            if has_defined:
                raise RuntimeError(
                    "If any step in the pipeline has defined Label, "
                    "only fit(X) is allowed or the training becomes "
                    "ambiguous.")

    @trace
    def get_feature_contributions(self, X, top=10, bottom=10, verbose=0, 
                                  as_binary_data_stream=False, **params):
        """
        Calculates observation level feature contributions. Returns dataframe
        with raw data, predictions, and feature contributiuons for each
        prediction. Feature contributions are not supported for transforms, so
        make sure that the last step in a pipeline is a model. Feature
        contriutions are supported for the following models:

        * Regression:

            * OrdinaryLeastSquaresRegressor
            * FastLinearRegressor
            * OnlineGradientDescentRegressor
            * PoissonRegressionRegressor
            * GamRegressor
            * LightGbmRegressor
            * FastTreesRegressor
            * FastForestRegressor
            * FastTreesTweedieRegressor

        * Binary Classification:

            * AveragedPerceptronBinaryClassifier
            * LinearSvmBinaryClassifier
            * LogisticRegressionBinaryClassifier
            * FastLinearBinaryClassifier
            * SgdBinaryClassifier
            * SymSgdBinaryClassifier
            * GamBinaryClassifier
            * FastForestBinaryClassifier
            * FastTreesBinaryClassifier
            * LightGbmBinaryClassifier

        * Ranking:

            * LightGbmRanker

        :param X: {array-like [n_samples, n_features],
            :py:class:`nimbusml.FileDataStream` }
        :param top: the number of positive contributions with highest magnitude
            to report.
        :param bottom: The number of negative contributions with highest
            magnitude to report.
        :return: dataframe of containing the raw data, predicted label, score,
            probabilities, and feature contributions.
        """
        self.verbose = verbose

        if not self._is_fitted:
            raise ValueError(
                "Model is not fitted. Train or load a model before test().")

        if len(self.steps) > 0:
            last_node = self.last_node
            if last_node.type == 'transform':
                raise ValueError(
                    "Pipeline needs a trainer as last step for test()")

        X, y_temp, columns_renamed, feature_columns, label_column, \
            schema, weights, weight_column = self._preprocess_X_y(X)

        all_nodes = []
        inputs = dict([('data', ''), ('predictor_model', self.model)])
        if isinstance(X, FileDataStream):
            importtext_node = data_customtextloader(
                input_file="$file",
                data="$data",
                custom_schema=schema.to_string(
                    add_sep=True))
            all_nodes = [importtext_node]
            inputs = dict([('file', ''), ('predictor_model', self.model)])

        score_node = transforms_datasetscorer(
            data="$data",
            predictor_model="$predictor_model",
            scored_data="$scoredvectordata")

        fcc_node = transforms_featurecontributioncalculationtransformer(
            data="$scoredvectordata",
            predictor_model="$predictor_model",
            output_data="$output_data",
            top=top,
            bottom=bottom,
            normalize=True)
        
        all_nodes.extend([score_node, fcc_node])

        outputs = dict(output_data="")

        graph = Graph(
            inputs,
            outputs,
            as_binary_data_stream,
            *all_nodes)

        class_name = type(self).__name__
        method_name = inspect.currentframe().f_code.co_name
        telemetry_info = ".".join([class_name, method_name])

        try:
            (out_model, out_data, out_metrics) = graph.run(
                X=X,
                random_state=self.random_state,
                model=self.model,
                verbose=verbose,
                telemetry_info=telemetry_info,
                **params)
        except RuntimeError as e:
            raise e

        return out_data

    @trace
    def _predict(self, X, y=None,
                 evaltype='auto', group_id=None,
                 weight=None,
                 verbose=0,
                 as_binary_data_stream=False, **params):
        """
        Apply transforms and test with the final estimator, return metrics
        """
        # start the clock!
        start_time = time.time()
        self.verbose = verbose

        if not self._is_fitted:
            raise ValueError(
                "Model is not fitted. Train or load a model before test("
                ").")

        if y is not None:
            if len(self.steps) > 0:
                last_node = self.last_node
                if last_node.type == 'transform':
                    raise ValueError(
                        "Pipeline needs a trainer as last step for test()")

        X, y_temp, columns_renamed, feature_columns, label_column, \
            schema, weights, weight_column = self._preprocess_X_y(
                X, y, w=weight
            )

        if (not isinstance(y, (str, tuple))) or (
                isinstance(X, DataFrame) and isinstance(y, (str, tuple))):
            y = y_temp

        all_nodes = []
        inputs = dict([('data', ''), ('predictor_model', self.model)])
        if isinstance(X, FileDataStream):
            importtext_node = data_customtextloader(
                input_file="$file",
                data="$data",
                custom_schema=schema.to_string(
                    add_sep=True))
            all_nodes = [importtext_node]
            inputs = dict([('file', ''), ('predictor_model', self.model)])

        score_node = transforms_datasetscorer(
            data="$data",
            predictor_model="$predictor_model",
            scored_data="$scoredVectorData")
        all_nodes.extend([score_node])

        if (evaltype in ['binary', 'multiclass']) or \
           (hasattr(self, 'steps')
            and self.steps is not None
            and len(self.steps) > 0
            and self.last_node.type == 'classifier'):

            select_node = transforms_scorecolumnselector(
                data="$scoredVectorData",
                output_data="$scoreColumnsOnlyData", score_column="Score")
            convert_label_node = \
                transforms_predictedlabelcolumnoriginalvalueconverter(
                    data="$scoreColumnsOnlyData",
                    predicted_label_column="PredictedLabel",
                    output_data="$output_data")
            all_nodes.extend([select_node, convert_label_node])
        else:
            select_node = transforms_scorecolumnselector(
                data="$scoredVectorData",
                output_data="$output_data", score_column="Score")
            all_nodes.extend([select_node])

        if y is not None:
            evaluate_nodes = self._evaluation_infer(
                evaltype, label_column, group_id, **params)
            for node in evaluate_nodes:
                all_nodes.extend([node])
            output_scores = '' if params.get(
                'output_scores', False) else '<null>'
            outputs = OrderedDict(
                [('output_metrics', ''), ('output_data', output_scores)])
        else:
            outputs = dict(output_data="")

        graph = Graph(
            inputs,
            outputs,
            as_binary_data_stream,
            *all_nodes)

        class_name = type(self).__name__
        method_name = inspect.currentframe().f_code.co_name
        telemetry_info = ".".join([class_name, method_name])

        try:
            (out_model, out_data, out_metrics) = graph.run(
                X=X,
                y=y,
                random_state=self.random_state,
                model=self.model,
                verbose=verbose,
                telemetry_info=telemetry_info,
                **params)
        except RuntimeError as e:
            self._run_time = time.time() - start_time
            raise e

        if y is not None:
            # We need to fix the schema for ranking metrics
            if evaltype == 'ranking':
                out_metrics = self._fix_ranking_metrics_schema(out_metrics)

        # stop the clock
        self._run_time = time.time() - start_time
        self._write_csv_time = graph._write_csv_time
        return out_data, out_metrics

    def _extract_classes(self, y):
        if ((len(self.steps) > 0) and
            (self.last_node.type in ['classifier', 'anomaly']) and
            (y is not None) and
            (not isinstance(y, (str, tuple)))):

            unique_classes = unique_labels(y)
            if len(unique_classes) < 2:
                raise ValueError(
                    "Classifier can't train when only one class is "
                    "present.")
            self._add_classes(unique_classes)

    def _extract_classes_from_headers(self, headers):
        classes = [x.replace('Score.', '') for x in headers]
        classes = np.array(classes).astype(self.last_node.classes_.dtype)
        self._add_classes(classes)

    def _add_classes(self, classes):
        # Create classes_ attribute similar to scikit
        # Add both to pipeline and ending classifier
        self.classes_ = classes
        self.last_node.classes_ = classes

    @trace
    def predict(self, X, verbose=0, as_binary_data_stream=False, **params):
        """
        Predict based on the input data

        :param X: {array-like [n_samples, n_features],
            :py:class:`nimbusml.FileDataStream` }
        """
        out_data, out_metrics = self._predict(
            X, verbose=verbose,
            as_binary_data_stream=as_binary_data_stream, **params)
        return out_data

    @trace
    def predict_proba(self, X, verbose=0, **params):
        """
        Apply transforms and predict probabilities

        :param X: {array-like [n_samples, n_features],
            :py:class:`nimbusml.FileDataStream` }

        :return: array, shape = [n_samples, n_classes]
        """
        if hasattr(self, 'steps') and len(self.steps) > 0:
            last_node = self.last_node
            last_node._check_implements_method('predict_proba')

        scores, _ = self._predict(X, verbose=verbose, **params)

        # REVIEW: Consider adding an entry point that extracts the
        # probability column instead.
        # This will enable returning as a binary IDV, as well as will be
        #  more efficient as it will not pass the unnecessary columns
        # from ML.NET to Python.
        # for binary classifiers
        if 'Probability' in scores.columns:
            positive_class_probs = \
                scores.loc[:, scores.columns == 'Probability'].values
            negative_class_probs = 1 - positive_class_probs
            return np.column_stack(
                (negative_class_probs, positive_class_probs))

        # for multiclass, scores are probabilities
        pcols = [i for i in scores.columns if i.startswith('Score.')]
        if len(pcols) > 0:
            self._extract_classes_from_headers(pcols)
            return scores.loc[:, pcols].values

        raise ValueError(
            "Predictor did not generate probabilites." +
            scores.columns)

    @trace
    def decision_function(self, X, verbose=0, **params):
        """
        Apply transforms and generate decision values

        :param X: {array-like [n_samples, n_features],
            :py:class:`nimbusml.FileDataStream` }

        :return: array, shape=(n_samples,) if n_classes == 2 else (
            n_samples, n_classes)
        """
        if hasattr(self, 'steps') and len(self.steps) > 0:
            last_node = self.last_node
            last_node._check_implements_method('decision_function')

        scores, _ = self._predict(X, verbose=verbose, **params)

        # REVIEW: Consider adding an entry point that extracts the score
        #  column instead.
        # This will enable returning as a binary IDV, as well as will be
        #  more efficient as it will not
        # pass the unnecessary columns from ML.NET to Python.
        # for binary classifiers
        scols = [i for i in scores.columns if i.startswith('Score.')]

        # for binary classifiers or multiclass with n_classes == 2
        if 'Score' in scores.columns:
            return scores.loc[:, 'Score'].values
        elif len(scols) == 2:
            scol = scols[-1]
            return scores.loc[:, scol].values

        # for multiclass with n_classes > 2
        if len(scols) > 2:
            self._extract_classes_from_headers(scols)
            return scores.loc[:, scols].values

        raise ValueError(
            "Predictor did not generate scores." + scores.columns)

    @trace
    def test(
            self,
            X,
            y=None,
            evaltype='auto',
            group_id=None,
            weight=None,
            verbose=0,
            output_scores=False,
            as_binary_data_stream=False,
            **params):
        """
        Return both predictions and performance metrics. For more
        details please
        refer to
        `Metrics </nimbusml/Metrics>`_.

        :param X: {array-like [n_samples, n_features],
            :py:class:`nimbusml.FileDataStream` }
        :param y: {array-like [n_samples]}

        :param evaltype: the evaluation type for the problem, can be {
            'binary', 'multiclass', 'regression', 'cluster', 'anomaly',
            'ranking'}. The default is 'auto'. If model is loaded using the
            load_model() method, evaltype cannot be 'auto', and therefore
            must be explicitly specified.
        :param group_id: the column name for group_id for ranking problem
        :param weight: the column name for the weight column for each
            sample
        :param output_scores: if set to True will return raw scores,
            otherwise None
            in the returned tuple.
        :return: tuple (dataframe of evaluation metrics, dataframe of
            scores). If scores are
            required, set `output_scores`=True, otherwise None is
            returned by default.
        """

        params['output_scores'] = output_scores
        if not self._is_fitted:
            raise ValueError(
                "Model is not fitted. Train or load a model before test("
                ").")

        errmsg = "'y' column cannot be inferred from pipeline. Please " \
                 "specify 'y' explicitly"
        if len(self.steps) > 0:
            last_node = self.last_node
            if last_node.type == 'transform':
                raise ValueError(
                    "Pipeline needs a trainer as last step for test()")
            if y is None:
                y = self.last_node.label_column_name_
        elif y is None:
            raise ValueError(errmsg)

        try:
            inputs = self._get_last_node_roles(X, y, **params)
            if y is None:
                if Role.Label not in inputs:
                    raise ValueError(errmsg)
                y = inputs[Role.Label]

            weight = weight if weight is not None else inputs.get(
                Role.Weight)
            group_id = group_id if group_id is not None else inputs.get(
                Role.GroupId)
            if group_id is None:
                if hasattr(last_node, 'row_group_column_name_'):
                    group_id = last_node.row_group_column_name_
        # if model was loaded using load_model, no nodes present
        except TypeError:
            pass

        if group_id is not None:
            # If group_id is passed as parameter, we set evaltype 'auto' to
            # 'ranking'
            if evaltype == 'auto':
                evaltype = 'ranking'

            # Check to make sure user has not used group_id with a
            # non-ranking
            # scenario
            if evaltype != 'ranking':
                raise ValueError(
                    "group_id invalid if evaltype !='ranking'.")

            if isinstance(group_id, DataFrame):
                # Do not move at the beginning: circular import
                from .internal.utils.dataframes import pd_concat
                if (group_id.shape[1]) != 1:
                    raise ValueError("group_id.shape[1] != 1")
                if (group_id.shape[0] != X.shape[0]):
                    raise ValueError(
                        "group_id, X length mismatch. %d vs %d" %
                        (group_id.shape[0], X.shape[0]))
                if (group_id.columns[0] in X.columns):
                    raise ValueError(
                        "column name for group '%s' already in X" %
                        group_id.columns[0])
                X.reset_index(inplace=True, drop=True)
                group_id.reset_index(inplace=True, drop=True)
                colnames = X.columns.append(group_id.columns)
                X = pd_concat([X, group_id.astype(np.uint32)], axis=1)
                X.columns = colnames
                groupcolname = group_id.columns[0]
                out_scores, out_metrics = self._predict(
                    X, y, evaltype,
                    group_id=groupcolname,
                    verbose=verbose,
                    as_binary_data_stream=as_binary_data_stream,
                    **params)
                return out_metrics, out_scores
            elif isinstance(group_id, (str, tuple)):
                pass
            else:
                raise ValueError("invalid group_id type'.")

        out_scores, out_metrics = self._predict(
            X, y, evaltype, group_id,
            weight, verbose=verbose,
            as_binary_data_stream=as_binary_data_stream,
            **params
        )
        return out_metrics, out_scores

    @trace
    def transform(
            self,
            X,
            y=None,
            verbose=0,
            as_binary_data_stream=False,
            **params):
        """
        Apply transforms

        :param X: {array-like [n_samples, n_features],
            :py:class:`nimbusml.FileDataStream` }
        :param y: {array-like [n_samples]}

        """
        # start the clock!
        start_time = time.time()
        self.verbose = verbose

        if not self._is_fitted:
            raise ValueError(
                "Model is not fitted. Train or load a model before test("
                ").")

        if y is not None:
            if len(self.steps) > 0:
                last_node = self.last_node
                if last_node.type == 'transform':
                    raise ValueError(
                        "Pipeline needs a trainer as last step for test()")

        X, y_temp, columns_renamed, feature_columns, label_column, \
            schema, weights, weight_column = self._preprocess_X_y(X, y)

        if not isinstance(y, (str, tuple)):
            y = y_temp

        all_nodes = []

        inputs = dict([('data', ''), ('transform_model', self.model)])
        if isinstance(X, FileDataStream):
            importtext_node = data_customtextloader(
                input_file="$file",
                data="$data",
                custom_schema=schema.to_string(
                    add_sep=True))
            all_nodes = [importtext_node]
            inputs = dict([('file', ''), ('transform_model', self.model)])

        apply_node = models_datasettransformer(
            data="$data",
            transform_model="$transform_model",
            output_data="$output_data")

        all_nodes.extend([apply_node])

        graph = Graph(
            inputs,
            dict(output_data=""),
            as_binary_data_stream,
            *all_nodes)

        class_name = type(self).__name__
        method_name = inspect.currentframe().f_code.co_name
        telemetry_info = ".".join([class_name, method_name])
        max_slots = params.pop('max_slots', -1)

        try:
            (out_model, out_data, out_metrics) = graph.run(
                X=X,
                random_state=self.random_state,
                model=self.model,
                verbose=verbose,
                max_slots=max_slots,
                telemetry_info=telemetry_info,
                **params)
        except RuntimeError as e:
            self._run_time = time.time() - start_time
            raise e

        # stop the clock
        self._run_time = time.time() - start_time
        self._write_csv_time = graph._write_csv_time
        return out_data

    @trace
    def summary(self, verbose=0, **params):
        """
        Return summary for fitted model.

        Example:
           .. literalinclude:: /../nimbusml/examples/Pipeline.py
                  :language: python
        """
        if hasattr(self,
                   'model_summary') and self.model_summary is not None:
            return self.model_summary

        # start the clock!
        start_time = time.time()
        self.verbose = verbose

        if not self._is_fitted:
            raise ValueError(
                "Model is not fitted. Train or load a model before "
                "summary().")

        # check last step is predictor in case there are steps in pipeline
        # importing here to break cycle import cycle dependency between
        # pipeline and base_predictor
        from .base_predictor import BasePredictor
        if len(self.steps) > 0 and not isinstance(
                self.last_node, BasePredictor):
            raise ValueError(
                "Summary is availabe only for predictor types, instead "
                "got " +
                self.last_node.type)

        all_nodes = []
        inputs = dict([('predictor_model', self.model)])

        summary_node = models_summarizer(
            predictor_model="$predictor_model",
            summary="$output_data")
        all_nodes.extend([summary_node])

        outputs = dict(output_data="")

        graph = Graph(
            inputs,
            outputs,
            False,
            *all_nodes)

        class_name = type(self).__name__
        method_name = inspect.currentframe().f_code.co_name
        telemetry_info = ".".join([class_name, method_name])

        try:
            (_, summary_data, _) = graph.run(
                X=None,
                y=None,
                random_state=self.random_state,
                model=self.model,
                verbose=verbose,
                is_summary=True,
                telemetry_info=telemetry_info,
                **params)
        except RuntimeError as e:
            self._run_time = time.time() - start_time
            raise e

        self._validate_model_summary(summary_data)
        self.model_summary = summary_data

        # stop the clock
        self._run_time = time.time() - start_time
        self._write_csv_time = graph._write_csv_time
        return self.model_summary

    @trace
    def _validate_model_summary(self, model_summary):
        """
        Validates model summary has correct format

        :param model_summary: model summary dataframes

        """
        if not isinstance(model_summary, (DataFrame)):
            raise TypeError(
                "Unexpected type {0} for model_summary, type DataFrame "
                "is expected ".format(
                    type(model_summary)))

        col_names = [
            'Bias',
            'ClassNames',
            'Coefficients',
            'PredictorName',
            'Summary',
            'VectorName'
        ]

        col_name_prefixes = [
            'Weights',
            'Gains',
            'Support vectors.',
            'VectorData'
        ]

        for col in model_summary.columns:
            if col in col_names:
                pass
            elif any([col.startswith(pre) for pre in col_name_prefixes]):
                pass
            else:
                raise TypeError(
                    "Unsupported '{0}' column is in model_summary".format(
                        col))

    @trace
    def save_model(self, dst):
        """
        Save model to file. For more details, please refer to
        `load/save model </nimbusml/loadsavemodels>`_

        :param dst: filename to be saved with

        """
        if self.model is not None:
            if os.path.isfile(self.model):
                copyfile(self.model, dst)

    @trace
    def load_model(self, src):
        """
        Load model from file. The model can be generated from ML.NET in
        .zip format.
        For more details, please refer to
        `load/save model </nimbusml/loadsavemodels>`_

        :param dst: source filename to be loaded

        """
        if not os.path.isfile(src):
            raise ValueError("file not found %s" % src)
        self.model = src
        self.steps = []

    def __getstate__(self):
        odict = {'export_version': 1}

        if hasattr(self, 'steps'):
            odict['steps'] = self.steps

        if (hasattr(self, 'model') and 
            self.model is not None and
            os.path.isfile(self.model)):

            with open(self.model, "rb") as f:
                odict['modelbytes'] = f.read()

        return odict

    def __setstate__(self, state):
        self.steps = []
        self.model = None
        self.random_state = None

        for k, v in state.items():
            if k not in {'modelbytes', 'export_version'}:
                setattr(self, k, v)

        if state.get('export_version', 0) == 1:
            if 'modelbytes' in state:
                (fd, modelfile) = tempfile.mkstemp()
                fl = os.fdopen(fd, "wb")
                fl.write(state['modelbytes'])
                fl.close()
                self.model = modelfile

    @trace
    def score(
            self,
            X,
            y,
            evaltype='auto',
            group_id=None,
            weight=None,
            verbose=0,
            **params):
        """
        Return performance metrics for the corresponding problem

        :param X: {array-like [n_samples, n_features],
            :py:class:`nimbusml.FileDataStream` }
        :param y: {array-like [n_samples]}
        :param evaltype: the evaluation type for the problem, can be {
            'binary', 'multiclass', 'regression', 'cluster',
            'anomaly', 'ranking'}. The default is 'auto'.
            If model is loaded using the load_model() method,
            evaltype cannot be 'auto', and therefore must
            be explicitly specified.
        """

        metrics, scores = self.test(
            X, y, evaltype, group_id, weight, verbose=verbose,
            output_scores=False, **params)
        task_type = evaltype if evaltype is not 'auto' else \
            self._last_node_type()

        if 'binary' in task_type:
            return metrics['AUC'][0]
        elif 'multiclass' in task_type:
            return metrics['Accuracy(micro-avg)'][0]
        elif 'regress' in task_type:
            return metrics['R Squared'][0]
        elif 'cluster' in task_type:
            return metrics['NMI'][0]
        elif 'anomaly' in task_type:
            return metrics['AUC'][0]
        elif 'rank' in task_type:
            return metrics['NDCG@1'][0]
        else:
            raise ValueError(
                "cannot generate score for {0}).".format(task_type))


    @classmethod
    def combine_models(cls, *items, **params):
        """
        Combine the models of multiple pipelines, transforms
        and/or predictors in to a single model. The models are
        combined in the order they are seen.

        :param items: the fitted pipelines, transforms and/or
            predictors which contain the models to join.

        :param contains_predictor: Set to `True` if the
            last item contains or is a predictor. Set to
            `False` if `items` only contains transforms.
            The default is True.

        :return: A new Pipeline which is backed by a model that
            is the combination of all the models passed in
            through `items`.
        """
        if len(items) == 0:
            raise RuntimeError(
                'At least one transform, predictor'
                'or pipeline must be specified.')

        for item in items:
            if not item._is_fitted:
                raise RuntimeError(
                    'Item must be fitted before'
                    'models can be combined.')

        contains_predictor = params.get('contains_predictor', True)
        verbose = params.get('verbose', 0)

        get_model = lambda x: x.model if hasattr(x, 'model') else x.model_

        if len(items) == 1:
            return Pipeline(model=get_model(items[0]))

        start_time = time.time()

        nodes = []
        inputs = {}
        transform_models = []

        for index, item in enumerate(items[:-1], start=1):
            var_name = 'transform_model' + str(index)
            inputs[var_name] = get_model(item)
            transform_models.append("$" + var_name)

        if contains_predictor:
            inputs['predictor_model'] = get_model(items[-1])

            combine_models_node = transforms_manyheterogeneousmodelcombiner(
                transform_models=transform_models,
                predictor_model='$predictor_model',
                model='$output_model')
            nodes.append(combine_models_node)

        else:
            var_name = 'transform_model' + str(len(items))
            inputs[var_name] = get_model(items[-1])
            transform_models.append("$" + var_name)

            combine_models_node = transforms_modelcombiner(
                models=transform_models,
                output_model='$output_model')
            nodes.append(combine_models_node)

        outputs = dict(output_model="")

        graph = Graph(
            inputs,
            outputs,
            False,
            *nodes)

        class_name = cls.__name__
        method_name = inspect.currentframe().f_code.co_name
        telemetry_info = ".".join([class_name, method_name])

        try:
            (out_model, _, _) = graph.run(
                X=None,
                y=None,
                random_state=None,
                model=None,
                verbose=verbose,
                is_summary=False,
                telemetry_info=telemetry_info,
                no_input_data=True,
                **params)
        except RuntimeError as e:
            raise e

        pipeline = Pipeline(model=out_model)

        # stop the clock
        pipeline._run_time = time.time() - start_time
        pipeline._write_csv_time = graph._write_csv_time

        return pipeline

