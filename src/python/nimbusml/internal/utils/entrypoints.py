# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
"""
definition of classes for the entities in the entrypoint manifest.json
"""
import functools
import json
import os
import tempfile
from collections import OrderedDict
from enum import Enum

import pandas as pd
import six
from pandas import DataFrame
from scipy.sparse import csr_matrix
from nimbusml.utils import signature

from .data_stream import DprepDataStream
from .data_stream import BinaryDataStream
from .data_stream import FileDataStream
from .dataframes import resolve_dataframe, resolve_csr_matrix, pd_concat, \
    resolve_output
from .utils import try_set, set_clr_environment_vars, get_clr_path, \
    get_mlnet_path, get_dprep_path
from ..libs.pybridge import px_call


class BridgeRuntimeError(RuntimeError):
    """
    Exception raises when the bridges fails to complete the task.
    """

    def __init__(self, msg, **kwargs):
        RuntimeError.__init__(self, msg)
        for k, v in kwargs.items():
            assert 'group_id' not in k
            setattr(self, k, v)


class Component(dict):
    """
    component
    """
    indent = 4
    sort_keys = True

    def __init__(self, name, settings, kind, desc=None):
        self.name = name
        self.settings = settings
        self.kind = kind
        self.desc = desc
        # NOTE: fill super (i.e., dict) for json.dumps to
        # encode.
        super(Component, self).__init__(Name=name, Settings=settings)

    def __str__(self):
        return json.dumps(
            self.to_dict(),
            indent=Component.indent,
            sort_keys=Component.sort_keys)

    def to_dict(self):
        """
        convert to dictionary
        """
        return dict(Name=self.name, Settings=self.settings)

    # the deep parameter is added to be consistent with sklearn for
    # GridSearchCV
    def get_params(self, deep=True):
        "Scikit-learn API, returns all parameters."

        sig = signature(self.__class__.__init__)
        params = [(p if p != 'columns' else '_columns', p)
                  for p in sig.parameters if p not in ('self', 'params')]
        res = {p: getattr(self, att)
               for att, p in params if hasattr(self, att)}
        if hasattr(self, "_columns") and isinstance(self._columns, dict):
            res['columns'] = self._columns

        return res


class EntryPoint:
    """
    entrypoint
    """
    indent = 4
    sort_keys = True

    def __init__(
            self,
            name,
            inputs,
            outputs,
            input_variables=None,
            output_variables=None,
            desc=None):
        self.name = name
        self.inputs = inputs
        self.outputs = outputs
        self.input_variables = input_variables
        self.output_variables = output_variables
        self.desc = desc
        # Nested entrypoints contain other entrypoints as Nodes inside them
        self.inner_nodes = inputs.get('Nodes')

    def __str__(self):
        return json.dumps(
            self.to_dict(),
            indent=EntryPoint.indent,
            sort_keys=EntryPoint.sort_keys)

    def to_dict(self):
        """
        convert to dictionary
        """
        if self.inner_nodes:
            expanded_nodes = [n.to_dict() for n in self.inner_nodes]
            expanded_inputs = self.inputs.copy()
            expanded_inputs['Nodes'] = expanded_nodes
            return dict(
                Name=self.name,
                Inputs=expanded_inputs,
                Outputs=self.outputs)
        else:
            return dict(
                Name=self.name,
                Inputs=self.inputs,
                Outputs=self.outputs)


def _get_temp_file(suffix=None):
    fd, file_name = tempfile.mkstemp(suffix=suffix)
    fl = os.fdopen(fd, 'w')
    fl.close()
    return file_name


class Graph(EntryPoint):
    """
    graph
    """

    @staticmethod
    def _check_nodes(nodes, node_path=''):
        '''Recursively check if all the graph nodes are entrypoints'''
        if nodes is None:
            # base case for recursive checking
            return

        for i, obj in enumerate(nodes):
            node_path = node_path + '[{}]'.format(i)
            if not isinstance(obj, EntryPoint):
                error_msg = "Node {} is not an entrypoint but {}.".format(
                    node_path, type(obj))
                raise TypeError(error_msg)

            Graph._check_nodes(obj.inner_nodes, node_path)

    def __init__(
            self,
            inputs=None,
            outputs=None,
            output_binary_data_stream=False,
            *nodes):
        Graph._check_nodes(nodes)

        # A variable is a string beginning with a $, and it signifies that a
        # specific input or output of a node needs to be loaded or saved
        # during the graph execution.
        input_variables = functools.reduce(
            set.union, [node.input_variables for node in nodes])
        output_variables = functools.reduce(
            set.union, [node.output_variables for node in nodes])

        # Each variable can appear in as many node inputs as desired,
        # but it can only appear as at most one node output.
        if len(set(output_variables)) != len(output_variables):
            raise ValueError("duplicated output_variables are not allowed.")

        self.input_variables = set(input_variables) - set(output_variables)
        self.output_variables = set(output_variables) - set(input_variables)

        self.set_inputs(inputs)
        self._set_outputs(outputs)

        self.nodes = nodes
        self._write_csv_time = 0
        self._output_binary_data_stream = output_binary_data_stream

    def __iter__(self):
        return iter(self.nodes)

    def set_inputs(self, inputs):
        """
        set graph inputs
        """
        if inputs is not None:
            if not isinstance(inputs, dict):
                raise ValueError(
                    "inputs to the graph should be of type 'dict'.")

            # A graph input is a variable that doesn't appear as any output.
            # All graph inputs must be provided for the graph to run.
            # The 'execgraph' command accepts file paths as a way to provide
            # graph inputs.
            graph_inputs = self.input_variables
            if {input.replace('$', '') for input in
                    graph_inputs} != set(inputs.keys()):
                raise ValueError(
                    "inputs to the graph do not match graph_inputs.")

        self.inputs = inputs

    def _set_outputs(self, outputs):
        """
        set graph outputs
        """
        if outputs is not None:
            if not isinstance(outputs, dict):
                raise ValueError(
                    "outputs from the graph should be of type 'dict'.")

            # A graph output is a variable that appears as an output and
            # needs to be saved even after the graph is finished running.
            # The 'execgraph' command writes the requested outputs to the
            # specified file paths.
            graph_outputs = self.output_variables
            if {output.replace('$', '')
                    for output in graph_outputs} != set(outputs.keys()):
                raise ValueError(
                    "outputs from the graph do not match graph_outputs.")

        self.outputs = outputs

    def to_dict(self):
        res = [node.to_dict() for node in self.nodes]
        if self.inputs is not None or self.outputs is not None:
            res = dict(nodes=res, inputs=self.inputs, outputs=self.outputs)
        return res

    @property
    def nimbusml_runnable_graph(self):
        return str(self).replace(
            '"inputs"',
            '"Inputs"').replace(
            '"outputs"',
            '"Outputs"').replace(
            '"nodes"',
            '"Nodes"')

    def run(
            self,
            X,
            y=None,
            seed=None,
            parallel=None,
            max_slots=-1,
            random_state=None,
            verbose=1,
            **params):
        """
        run graph
        """
        code = ""
        if parallel is not None:
            if isinstance(parallel, six.integer_types):
                code += "parallel = {} ".format(parallel)
            else:
                raise TypeError("parallel is not of 'int' type.")
        if seed is not None:
            if isinstance(seed, six.integer_types):
                code += "seed = {} ".format(seed)
            else:
                raise TypeError("seed is not of 'int' type.")
        if parallel is not None:
            if isinstance(parallel, six.integer_types):
                code += "parallel = {} ".format(parallel)
            else:
                raise TypeError("parallel is not of 'int' type.")
        if max_slots is not None:
            if isinstance(max_slots, six.integer_types):
                code += "maxSlots = {} ".format(max_slots)
            else:
                raise TypeError("max_slots is not of 'int' type.")

        if params.get("dryrun") is not None:
            ret = 'graph = {%s} %s' % (str(self), code)
        else:
            ret = self.idv_bridge(X, y, code, random_state, verbose, **params)
        return ret

    def _try_call_bridge(
            self,
            px_call,
            call_parameters,
            code,
            verbose,
            concatenated,
            output_modelfilename):
        try:
            ret = px_call(call_parameters)
        except RuntimeError as e:
            if verbose:
                vars = '?'
                if "data" in call_parameters:
                    if isinstance(
                            call_parameters.get(
                                "data",
                                None),
                            pd.DataFrame):
                        df = call_parameters["data"]
                        vars = "type={0} shape={1} columns={2}".format(
                            type(df), df.shape, df.columns)
                    elif isinstance(call_parameters.get("data", None),
                                    OrderedDict):
                        od = call_parameters["data"]
                        vars = "type={0} keys={1}".format(
                            type(od), ','.join(od))
                if isinstance(verbose, six.integer_types) and verbose >= 2:
                    raise BridgeRuntimeError(
                        "{0}.\n--CODE--\n{1}\n--GRAPH--\n{2}\n--DATA--\n{3}"
                        "\n--\nconcatenated={4}".format(
                            str(e), code, str(self), vars, concatenated),
                        model=output_modelfilename)
                else:
                    raise BridgeRuntimeError(
                        str(e), model=output_modelfilename)
            else:
                raise e
        return ret

    def _get_separator(self):
        node = self.nodes[0]
        inputs = getattr(node, 'inputs', None)
        if inputs is None:
            return None
        sch = inputs.get('CustomSchema', None)
        if sch is None:
            return None
        pieces = [_ for _ in sch.split(' ') if 'sep=' in _]
        if len(pieces) == 0:
            return None
        return pieces[0].replace("sep=", "").strip()

    def idv_bridge(self, X, y, code, random_state=None, verbose=1, **params):
        output_modelfilename = None
        output_metricsfilename = None
        out_metrics = None

        # Ideally, idv_bridge shouldn't care if it's running CV
        # or a regular pipeline. That required changing the idv_bridge to be
        # more flexible (e.g. changing return value, changing input
        # structure, etc.) In my first attempt, this approach caused
        # unintended test-failures that were hard to root cause. To eliminate
        # this side-effect, in the second attempt, I'm keeping the code path
        # for regular pipeline exactly the same as before.
        # The cv flag only applies to CV pipelines, so there shouldn't be
        # any side-effect on non-CV tests.
        cv = params.get('is_cv')

        # checks whether this is a model summary call
        summary = params.get('is_summary')

        try:
            concatenated = False
            call_parameters = {}
            if isinstance(X, DataFrame):

                def remove_multi_level_index(c):
                    if isinstance(c, (str, six.text_type)):
                        return c
                    elif isinstance(c, tuple):
                        return '.'.join(c)
                    else:
                        raise TypeError(
                            "Unexpected type {0} for a column name.".format(
                                type(c)))

                if y is not None and isinstance(y, DataFrame):
                    data = pd_concat([X, y], axis=1, join='inner')
                else:
                    # copy is mandatory here otherwise the order
                    # of the results might not be the same depending
                    # on the fact the dataframe is a view or not.
                    data = X.copy()
                # Removes Multi-Level index
                data.columns = [
                    remove_multi_level_index(c) for c in data.columns]
                call_parameters["data"] = resolve_dataframe(data)
                if y is not None:
                    concatenated = True
            elif isinstance(X, csr_matrix):
                call_parameters["data"] = resolve_csr_matrix(X, y)
                if y is not None:
                    concatenated = True
            elif isinstance(X, FileDataStream):
                self.inputs['file'] = X.filename
            elif isinstance(X, BinaryDataStream) or isinstance(X, DprepDataStream):
                if 'input_data' in self.inputs:
                    self.inputs['input_data'] = X._filename
                elif 'data' in self.inputs:
                    self.inputs['data'] = X._filename
            elif not (summary or params.get('no_input_data')):
                raise RuntimeError(
                    "data should be a dataframe, FileDataStream or DataView")

            if cv:
                output_types = params['output_types']
                self._set_file_outputs(output_types)
                params.pop('is_cv')
                params.pop('output_types')
            else:
                # set graph output model to temp file
                if 'output_model' in self.outputs:
                    output_modelfilename = _get_temp_file(suffix='.model.bin')
                    self.outputs['output_model'] = output_modelfilename

                # set graph output metrics to temp file
                if 'output_metrics' in self.outputs:
                    output_metricsfilename = _get_temp_file(suffix='.txt')
                    self.outputs['output_metrics'] = output_metricsfilename

                if 'output_data' in self.outputs and \
                        self._output_binary_data_stream:
                    output_idvfilename = _get_temp_file(suffix='.idv')
                    self.outputs['output_data'] = output_idvfilename

            # set graph file for debuggings
            if verbose > 0:
                # graph_id will allow for adding descriptive info to the graph
                # file name to be able to match graphs with different runs.
                graph_suffix = params.get('graph_id', '') + '.graph.txt'
                params.pop('graph_id', None)
                input_graphfilename = _get_temp_file(suffix=graph_suffix)
                with open(input_graphfilename, 'w') as f:
                    f.write(self.nimbusml_runnable_graph)

            call_parameters['verbose'] = try_set(verbose, False, six.integer_types)
            call_parameters['graph'] = try_set(
                'graph = {%s} %s' %
                (str(self), code), False, str)
            
            # Set paths to .NET Core CLR, ML.NET and DataPrep libs
            set_clr_environment_vars()
            call_parameters['dotnetClrPath'] = try_set(get_clr_path(), False, str)
            call_parameters['mlnetPath'] = try_set(get_mlnet_path(), False, str)
            call_parameters['dprepPath'] = try_set(get_dprep_path(), False, str)

            if random_state:
                call_parameters['seed'] = try_set(random_state, False, six.integer_types)
            ret = self._try_call_bridge(
                px_call,
                call_parameters,
                code,
                verbose,
                concatenated,
                output_modelfilename)

            out_data = resolve_output(ret)
            # remove label column from data
            if out_data is not None and concatenated:
                out_columns = list(out_data.columns)
                if hasattr(y, 'columns'):
                    y_column = y.columns[0]
                    if y_column in out_columns:
                        out_columns.remove(y_column)
                        out_data = out_data[out_columns]
            if output_metricsfilename:
                out_metrics = pd.read_csv(
                    output_metricsfilename,
                    sep='\t',
                    header=0,
                    error_bad_lines=False,
                    comment='#',
                    na_values='?')
            # todo: load & return model blob

            if cv:
                return self._process_graph_run_results(out_data)
            elif self._output_binary_data_stream:
                output = BinaryDataStream(output_idvfilename)
                return (output_modelfilename, output, out_metrics)
            else:
                return (output_modelfilename, out_data, out_metrics)
        finally:
            if cv:
                self._remove_temp_files()
            else:
                if output_modelfilename:
                    # os.remove(output_modelfilename)
                    pass
                if output_metricsfilename:
                    os.remove(output_metricsfilename)

    def _set_file_outputs(self, output_types):
        self.output_types = output_types

        for output_name, output_type in self.output_types.items():
            if output_type == GraphOutputType.ModelFile:
                self.outputs[output_name] = _get_temp_file(suffix='.model.bin')

            elif output_type == GraphOutputType.ModelArrayFile:
                self.outputs[output_name] = _get_temp_file(
                    suffix='.model_{0}.bin')

            elif output_type == GraphOutputType.TempFile:
                self.outputs[output_name] = _get_temp_file(suffix='.txt')

            elif output_type == GraphOutputType.BridgeReturnValue:
                self.outputs[output_name] = ''

            else:
                raise ValueError('Unsupported output type ', output_type)

    def _process_graph_run_results(self, bridge_df):
        results = {}

        for output_name, output_type in self.output_types.items():
            if output_type in [
                    GraphOutputType.ModelFile,
                    GraphOutputType.ModelArrayFile]:
                results[output_name] = self.outputs[output_name]

            elif output_type == GraphOutputType.TempFile:
                temp_file = self.outputs[output_name]
                results[output_name] = pd.read_csv(
                    temp_file, sep='\t', header=0, error_bad_lines=False,
                    comment='#')

            elif output_type == GraphOutputType.BridgeReturnValue:
                results[output_name] = bridge_df

        return results

    def _remove_temp_files(self):
        for output_name, output_type in self.output_types.items():
            # ModelArrayFile is a pattern to save each element of the array.
            # We can remove it without removing the actual saved models.
            if output_type in [
                    GraphOutputType.TempFile,
                    GraphOutputType.ModelArrayFile]:
                os.remove(self.outputs[output_name])


class GraphOutputType(Enum):
    BridgeReturnValue = 'bridge_return_value'
    TempFile = 'temp_file'
    ModelFile = 'model_file'
    ModelArrayFile = 'model_array_file'  # used for CV