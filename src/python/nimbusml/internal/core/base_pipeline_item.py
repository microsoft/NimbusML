# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
"""
Base class for all the items that can be used in a scikit or MicrosoftML
pipeline.
"""

__all__ = ["BasePipelineItem"]

import os
import tempfile
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from itertools import chain
from shutil import copyfile
from textwrap import wrap

import six
from nimbusml.utils import signature

from ..utils.data_roles import DataRoles, Role
from ..utils.data_stream import ViewBasePipelineItem, DataStream, \
    ViewDataStream
from ..utils.utils import trace


class BaseSignature:
    """
    Base class for signature.
    """

    def _use_role_except_feature(self):
        return False

    def _use_multi_output(self):
        return False

    def _use_single_input_as_string(self):
        return False

    def _use_only_one_output(self):
        return False

    def _use_no_output(self):
        return False

    def _use_unique_default_output_is_feature(self):
        return False

    def _check_inputs(self):
        pass

    def _check_outputs(self):
        pass


class DefaultSignature(BaseSignature):
    """
    Defines entrypoint expectations for inputs and outputs types
    (one column, two columns, ...).
    The design should be revisited but it was brought to
    be consistant with
    `scikit-sklean.base.clone
    <https://github.com/scikit-learn/scikit-learn/blob/a24c8b46/sklearn
    /base.py#L30>`_.
    """
    pass


class DefaultSignatureWithRoles(DefaultSignature):
    """
    Same as *DefaultSignature* but allows roles.
    """

    def _use_role_except_feature(self):
        return True


class NoOutputSignature(DefaultSignature):
    """
    Same as *DefaultSignature* but does not allow dictionaries
    (cannot create new columns).
    """

    def _use_no_output(self):
        return True

    def _check_inputs(self):
        """
        Some transforms can handle only one input columns defined as a list
        of one element: string or tuple (multi-index).
        """
        input = getattr(self, self._attr_input)
        if not isinstance(input, (str, tuple, list)):
            raise TypeError(
                "Unexpected type {0} for input, type str or list is "
                "expected (input_schema='{1}')".format(
                    type(input), self._use_input_schema()))

    def _check_outputs(self):
        """
        Checks there is no output.
        """
        if hasattr(self, '_attr_output'):
            output = getattr(self, self._attr_output)
            if output is not None:
                raise TypeError(
                    "Dictionaries are not allowed to specify input "
                    "columns (input_schema='{0}')".format(
                        self._use_input_schema()))


class MultiOutputsSignature(BaseSignature):
    """
    Some transforms can handle multiple inputs and outputs.
    input type is a list of lists, output type is a list.
    """

    def _use_multi_output(self):
        return True


class SingleInputAsStringSignature(BaseSignature):
    """
    Some transforms can handle only one input column defined as a string.
    """

    def _use_single_input_as_string(self):
        return True

    def _check_inputs(self):
        """
        Some transforms can handle only one input columns defiend as a list
        of one string.
        """
        input = getattr(self, self._attr_input)
        if not isinstance(input, (str, tuple)):
            raise TypeError(
                "Unexpected type {0} for input, type str is expected ("
                "input_schema={1})".format(
                    type(input), self._use_input_schema()))


class SingleOutputSignature(BaseSignature):
    """
    Some transforms can handle only one output column,
    input is defined as a list with one item.
    """

    def _use_only_one_output(self):
        return True

    def _check_inputs(self):
        """
        Some transforms can handle only one input columns defiend as a list
        of one string or one tuple.
        """
        input = getattr(self, self._attr_input)
        if not isinstance(input, list):
            raise TypeError(
                "Unexpected type {0} for input, type list is "
                "expected".format(
                    type(input)))
        if len(input) == 0:
            raise RuntimeError(
                "{0} needs one input at least".format(
                    type(self)))
        for i, inp in enumerate(input):
            if not isinstance(inp, (str, tuple)):
                raise TypeError(
                    "Unexpected type {0} for input {1}, type string is "
                    "expected\ninput_schema={2}\ninput={3}".format(
                        type(inp), i, self._use_input_schema(), input))


class SingleOutputSignatureWithRoles(SingleOutputSignature):
    """
    Same as *SingleOutputSignature* but allows roles.
    """

    def _use_role_except_feature(self):
        return True

    def _use_unique_default_output_is_feature(self):
        return True

    def _check_inputs(self):
        input = getattr(self, self._attr_input)
        for v in input:
            if isinstance(v, list):
                raise RuntimeError(
                    "Multi-ouput is not allowed - {0}".format(input))

    def _check_outputs(self):
        """
        Checks there is no output.
        """
        if hasattr(self, '_attr_output'):
            output = getattr(self, self._attr_output)
            if not isinstance(output, list):
                raise TypeError(
                    "Output should be one list not {0}".format(
                        type(output)))
            if len(output) != 1:
                raise RuntimeError(
                    "Output should contain only one output not {0} ("
                    "use_schema='{1}')".format(
                        output, self._use_input_schema()))
            if not isinstance(output[0], (str, tuple)):
                raise RuntimeError(
                    "Output should contain only one output defined as a "
                    "string not {0} (use_schema='{1}')".format(
                        output, self._use_input_schema()))


class EqualInputOutputSignature(SingleOutputSignature):
    """
    Each column becomes a new one. Number of inputs and outputs
    are the same.
    """

    def _check_inputs(self):
        input = getattr(self, self._attr_input)
        output = getattr(
            self, self._attr_output) if hasattr(
            self, '_attr_output') else input
        if len(input) != len(output):
            raise RuntimeError(
                "Number of inputs and outputs should be equal for type {"
                "0} ({1} != {2})".format(
                    type(self), len(input), len(output)))


@six.add_metaclass(ABCMeta)
class BasePipelineItem():
    """
    Base class for all pipeline items.
    Derived classes can be used both, MicrosoftML and Scikit-Learn
    pipelines
    """

    _hidden_constructor_arguments = set(
        chain(['columns'], map(lambda s: s.lower(), DataRoles._allowed)))

    @trace
    def __init__(self, type=None, random_state=None, **params):
        # The consctuctor is usually called twice.
        # First time from BaseSomething like BaseTransform.
        # Second from internal classes.
        if hasattr(self, '_BasePipelineItem_already_called'):
            return
        self._BasePipelineItem_already_called = True
        if type is None:
            raise ValueError("Type must be defined.")
        self.type = type
        if 'schema' in params:
            raise RuntimeError("Schema not allowed.")
        if 'input' in params:
            raise RuntimeError("Input not allowed.")
        if 'output' in params:
            raise RuntimeError("Output not allowed.")
        if 'columns' in params and type != 'transform' and params[
                "columns"] is not None:
            raise RuntimeError(
                "Predictor use arguements feature, label to defined "
                "roles, argument columns is not allowed.")
        self.random_state = random_state
        # It assumes all columns are used as input.
        self.input = None
        # Default options for output columns. Depends on the model.
        self.output = None
        sig_params = signature(self._entrypoint).parameters
        self._allowed_roles = set(
            r for r in DataRoles._allowed if
            Role.to_attribute(r) in sig_params)
        # Basic checking on parameters.
        for k, v in params.items():
            if '_num_' in k and not isinstance(v, (int, float)):
                raise TypeError(
                    "Parameter '{0}' is not numeric but {1}.".format(
                        k, type(v)))
        self._handle_extra_syntax_parameters(params)

    def _handle_extra_syntax_parameters(self, params):
        """
        Handles extra parameters given to the constructor such as
        *columns* or a role.
        """

        # remove column_ for roles
        def clean_name(name):
            return DataRoles._allowed_attr.get(name, name)

        set_params = set(map(clean_name, params))

        # Checks that extra parameters are allowed.
        sign = signature(self.__class__.__init__)
        allowed = set(sign.parameters)

        notin = set_params - allowed - \
            BasePipelineItem._hidden_constructor_arguments
        if len(notin) > 0:
            allowed = "\n".join(
                wrap(
                    ", ".join(
                        sorted(
                            filter(
                                lambda _: _ != 'self',
                                allowed)))))
            if len(notin) == 1:
                raise NameError(
                    "Parameter '{0}' is not allowed for class '{"
                    "1}'.\nAllowed: {2}".format(
                        list(
                            sorted(notin))[0],
                        self.__class__.__name__,
                        allowed))
            else:
                raise NameError(
                    "Parameters {0} are not allowed for class '{"
                    "1}'.\nAllowed: {2}".format(
                        sorted(notin), self.__class__.__name__, allowed))

        # Handles parameters columns.
        inputs = OrderedDict()
        cols = params.pop('columns', None)
        if cols:
            if isinstance(cols, dict):
                inputs.update(cols)
            else:
                self.set_inputs(cols, early=True)

        for role in DataRoles._allowed:
            name = DataRoles.to_attribute(role)
            if name in params:
                if cols is not None and role in cols and params[name] != \
                        cols[role]:
                    raise AttributeError(
                        "Attribute '{0}' is already set to '{1}', "
                        "cannot be replaced by '{2}'".format(
                            name, cols[role], params[name]))
                attr = DataRoles.to_attribute(role)
                if attr in allowed:
                    setattr(self, attr, params[name])
                else:
                    inputs[role] = params[name]
                del params[name]

        if len(inputs) > 0:
            self.set_inputs(inputs, early=True)

    def _getattr_role(self, role, all_args):
        """
        Picks the role value in self if not None, in all_args otherwise.
        role must be suffixed by `_column`.
        """
        value = getattr(self, role, None)
        if value is None:
            return all_args.get(role, None)
        return value

    @abstractmethod
    def _get_node(self, **params):
        """
        Composes an entrypoint node for the current item

        Returns
        -------
        The entrypoint node for the item.
        """
        pass

    def __getstate__(self):
        "Selects what to pickle."
        odict = self.__dict__.copy()
        odict['export_version'] = 1

        if hasattr(self, 'model_') and \
                self.model_ is not None and os.path.isfile(self.model_):
            with open(self.model_, "rb") as mfile:
                odict['modelbytes'] = mfile.read()
            del odict['model_']
        if hasattr(self, 'type'):
            odict['type'] = self.type
        return odict

    def __setstate__(self, state):
        "Restore a pickled object."
        for k, v in state.items():
            if k not in {'modelbytes', 'type', 'export_version'}:
                setattr(self, k, v)

        # Note: modelbytes and type were
        # added before export_version 1
        if 'modelbytes' in state:
            (fd, modelfile) = tempfile.mkstemp()
            fl = os.fdopen(fd, "wb")
            fl.write(state['modelbytes'])
            fl.close()
            self.model_ = modelfile
        if 'type' in state:
            self.type = state['type']
        else:
            raise AttributeError(
                "Type must be present in serialized object")

    def clone(self):
        "Clone the object."
        return self.__class__(**self.get_params())

    def get_params(self, deep=True):
        "Scikit-learn API with same params, returns all parameters."
        sig = signature(self.__class__.__init__)
        params = [(p if p != 'columns' else '_columns', p)
                  for p in sig.parameters if p not in ('self', 'params')]
        res = {p: getattr(self, att)
               for att, p in params if hasattr(self, att)}
        if hasattr(self, "_columns") and isinstance(self._columns, dict):
            res['columns'] = self._columns
        if self.type != "transform" and 'columns' in res:
            cols = res.pop('columns')
            if isinstance(cols, dict):
                for k, v in cols.items():
                    k2 = Role.to_attribute(k, "")
                    res[k2] = v
            else:
                res['feature'] = cols
        return res

    def get_roles_params(self):
        """
        Returns the subset of params related to roles.
        """
        pars = self.get_params()
        res = {}
        for role in DataRoles._allowed:
            attr = Role.to_attribute(role)
            if attr in pars and pars[attr]:
                res[attr] = pars[attr]
            if attr.endswith("_column"):
                attr = attr[:-7]
                if attr in pars and pars[attr]:
                    res[attr] = pars[attr]
        if "columns" in pars:
            res["columns"] = pars
        return res

    @trace
    def save_model(self, dst):
        """
        Save model to file. For more details, please refer to
        `load/save model </nimbusml/loadsavemodels>`_

        :param dst: filename to be saved with

        """
        if self.model_ is not None:
            if os.path.isfile(self.model_):
                copyfile(self.model_, dst)

    def __getitem__(self, cols):
        """
        Returns a View on this element restricted to the selected column.
        """
        return ViewBasePipelineItem(self, cols)

    def _use_input_schema(self):
        """
        Some transforms are using a different API to define inputs and
        outputs.
        (source, name) or (input, output). This methods returns True if
        the first one is used for this object.
        """
        if self._use_only_one_output():
            return 'so'
        if self._use_single_input_as_string():
            return 'si'
        sign = signature(self._entrypoint)
        for p in sign.parameters:
            if p == "source":
                return 'ns'
        return "io"

    def _check_roles(self):
        """
        Checks the consistency between defined roles and supported roles.
        """
        if not hasattr(self, '_entrypoint'):
            raise SystemExit(
                'One internal learner does not follow the new syntax.')
        params = signature(self._entrypoint).parameters
        for role in DataRoles._allowed:
            attr = DataRoles.to_attribute(role)
            if hasattr(self, attr) and getattr(self, attr) is not None and \
                    attr not in params:
                if role == Role.Label:
                    # warnings instead of an exception but we should
                    # really simplify the logic
                    # in experiment.py. The model should know which
                    # roles it supports.
                    # current code makes it difficult to guess.
                    # A minor modification in entrypoints.py should do the
                    # trick.
                    if self.type not in {"clusterer", "anomaly"} :
                        warnings.warn(
                            "Model '{0}' (type='{1}') does not support "
                            "role '{2}' (for developers, check "
                            "_allowed_roles is defined).".format(
                                type(self), self.type, role))
                else:
                    raise RuntimeError(
                        "Model '{0}' (type='{1}') does not support role "
                        "'{2}' (for developers, check _allowed_roles is "
                        "defined).".format(
                            type(self), self.type, role))

    def _use_role(self, name):
        """
        Tells if the transform or learner use role *name*.
        """
        return name in self._allowed_roles

    def __gt__(self, inp):
        """
        Operator ``>``
        """
        # return self._set_outputs(inp)
        raise NotImplementedError('Operator > is not supported')

    def __rshift__(self, to):
        """
        Operator ``>>``
        """
        # return self._set_outputs(to)
        raise NotImplementedError('Operator >> is not supported')

    def __or__(self, role_column):
        """
        Operator ``|``
        """
        raise NotImplementedError('Operator | is not supported')

    def __lt__(self, inp):
        """
        Equivalent to ``<<``.
        This operator has one drawback, it cannot be used with
        operator ``>`` to define the output due to python operator
        priorities.
        """
        return self.set_inputs(inp)

    def __lshift__(self, inp):
        """
        Change the input columns.
        This operator is part of the syntax used
        to define a pipeline (see
        `Columns </nimbusml/concepts/columns>`_).
        Operator ``<<`` is equivalent to parameter
        columns in the constructor. We update the
        attribute ``_columns``.
        """
        self._columns = inp
        return self.set_inputs(inp)

    def _add_attribute(self, name, value, input=False):
        if input:
            if not isinstance(value, list):
                raise TypeError(
                    "Unable to convert input into a list: {0}".format(
                        type(value)))

            if self._use_input_schema() == 'si':
                if len(value) != 1:
                    raise TypeError(
                        "Only one input column is allowed not {0}".format(
                            value))
                value = value[0]
            elif self._use_input_schema() == 'so':
                if isinstance(value[0], list):
                    if len(value) != 1:
                        raise TypeError(
                            "Only one output column is allowed not {"
                            "0}".format(
                                value))
                    value = value[0]

        current = getattr(self, name, None)
        if isinstance(current, list) and isinstance(value, list):
            current.extend(value)
        elif current is not None:
            raise AttributeError(
                "Attribute '{0}' is already set to '{1}', cannot be "
                "replaced by '{2}'".format(
                    name, current, value))
        else:
            setattr(self, name, value)

    def _set_outputs(self, to):
        """
        Change the outputs column.
        """
        if self._use_no_output():
            # Ignore.
            return self
        if self._use_input_schema() == "ns":
            attr = 'name'
            if isinstance(to, tuple):
                raise NotImplementedError(
                    '(name, type) not implemented yet.')
            if isinstance(to, (str, tuple)):
                self._add_attribute(attr, to)
            if not isinstance(getattr(self, attr), (str, tuple)):
                raise TypeError(
                    "Unable to convert name into a string: {0}".format(
                        type(
                            getattr(
                                self,
                                attr))))
        else:
            attr = 'output'
            if isinstance(to, (str, tuple)):
                self._add_attribute(attr, [to])
            else:
                self._add_attribute(attr, to)
            if not isinstance(getattr(self, attr), list):
                raise TypeError(
                    "Unable to convert output into a list: {0}".format(
                        type(
                            getattr(
                                self,
                                attr))))
        self._attr_output = attr
        self._check_outputs()
        return self

    def set_inputs(self, inp, early=False):
        """
        Change the input columns.

        :param inp: inputs (dictionary, list, str, tuple,
        see `Columns </nimbusml/concepts/columns>`_)
        :param early: set inputs from the constructor, object type is
        unknown
        """
        if isinstance(inp, (list, tuple, dict)):
            if len(inp) == 0:
                raise ValueError("inp is empty")
        elif inp in (None, ''):
            raise ValueError("inp is empty")
        if self.type not in ('transform', None):
            if isinstance(inp, dict):
                return self._set_role(inp)
            elif isinstance(inp, (str, tuple)):
                return self._set_role(inp, 'Feature')
        elif isinstance(inp, dict) and self._use_role_except_feature():
            inp = inp.copy()
            for k in DataRoles._allowed:
                if k in inp and self._use_role(k):
                    self._set_role(inp[k], role=k)
                    del inp[k]
            if len(inp) == 0:
                return self

        if not early and self.type != 'transform' and not self._use_role(
                'Feature'):
            raise RuntimeError(
                "This learner (type: '{0}') does not use role "
                "'Feature'.\nentrypoint={1}\nparams={2}".format(
                    self.type, self._entrypoint, ", ".join(
                        sorted(
                            signature(
                                self._entrypoint).parameters))))

        if self._use_input_schema() == "ns":
            # Couple source, name
            attr = 'source'
            if isinstance(inp, (str, tuple)):
                self._add_attribute(attr, inp)
                self._set_outputs(inp)
            elif isinstance(inp, list):
                if len(inp) != 1:
                    raise RuntimeError(
                        "Only one column is allowed for '{0}'.".format(
                            type(self)))
                self._add_attribute(attr, inp[0])
                self._set_outputs(inp[0])
            elif isinstance(inp, dict):
                if len(inp) != 1:
                    raise RuntimeError(
                        "Only one input is allowed for '{0}'.".format(
                            type(self)))
                key = list(inp.keys())[0]
                value = inp[key]
                if isinstance(value, list):
                    if len(value) != 1:
                        raise RuntimeError(
                            "Only one input is allowed for '{0}'.".format(
                                type(self)))
                    value = value[0]
                if not isinstance(value, (str, tuple)):
                    raise RuntimeError(
                        "'{0}' only accepts one input given as string or "
                        "tuple.".format(
                            type(self)))
                setattr(self, attr, value)
                self._set_outputs(key)
            else:
                self._add_attribute(attr, inp)
                raise NotImplementedError(
                    "Type '{0}' is not supported.".format(
                        type(inp)))
            if not isinstance(getattr(self, attr), (str, tuple)):
                raise TypeError(
                    "Unable to convert input into a string or a tuple: {"
                    "0}".format(
                        type(
                            getattr(
                                self,
                                attr))))

        elif self._use_multi_output():
            # Couple input, output
            attr = 'input'
            if isinstance(inp, dict):
                couples = [(k, v) for k, v in inp.items()]
                self._add_attribute(attr, [v for k, v in couples])
                self._set_outputs([k for k, v in couples])
            elif isinstance(inp, list):
                res = []
                is_string_or_tuple = False
                for i, v in enumerate(inp):
                    if isinstance(v, list) and not is_string_or_tuple:
                        res.append(v)
                    elif isinstance(v, (
                            DataStream, ViewDataStream)) and \
                            not is_string_or_tuple:
                        res.append([c.Name for c in inp.schema])
                    elif isinstance(v, (str, tuple)):
                        is_string_or_tuple = True
                        res.append(v)
                    else:
                        raise TypeError(
                            "Unexpected type for input {0}".format(i))
                if is_string_or_tuple:
                    self._add_attribute(attr, [res])
                else:
                    self._add_attribute(attr, res)
            else:
                self._add_attribute(attr, inp)
            if not isinstance(getattr(self, attr), list):
                raise TypeError(
                    "Unable to convert input into a list: {0}".format(
                        type(
                            getattr(
                                self,
                                attr))))
            for i, inp in enumerate(getattr(self, attr)):
                if not isinstance(inp, list):
                    raise TypeError(
                        "Input {0} is not a list but: {1}".format(
                            i, type(inp)))
        else:
            attr = 'input'
            if isinstance(inp, (str, tuple)):
                # tuple for MultiIndexColumn
                self._add_attribute(attr, [inp], input=True)
                self._set_outputs([inp])
            elif isinstance(inp, dict):
                couples = [(k, v) for k, v in inp.items()]
                self._add_attribute(attr, [v for k, v in couples],
                                    input=True)
                self._set_outputs([k for k, v in couples])
            elif isinstance(inp, list):
                self._add_attribute(attr, inp, input=True)
                if self._use_unique_default_output_is_feature() and len(
                        inp) != 1:
                    raise RuntimeError(
                        "The transform only allows only output, "
                        "use a dictionary to specify its name.")
                else:
                    self._set_outputs(inp)
            else:
                raise TypeError(
                    "Unexpected type for inp: {0}".format(
                        type(inp)))

        # Needed for learner. % is also used to define feature roles.
        if self.type in {'classifier', 'regressor',
                         'ranker', 'clustering', 'anomaly'}:
            self.feature_column_name = getattr(self, attr)
            if not isinstance(self.feature_column_name, (str, tuple)):
                if isinstance(self.feature_column_name, list):
                    if len(self.feature_column_name) == 1:
                        self.feature_column_name = self.feature_column_name[0]
                    else:
                        # Experiment will merge them.
                        # raise RuntimeError("Too many feature columns.
                        # Use ConcatTransform to merge them: "
                        #     " ConcatTransform() % {0} >
                        # Role.Feature".format(self.feature_column_name))
                        pass
                else:
                    raise TypeError(
                        "Feature column type is unexpected: {0}".format(
                            type(
                                self.feature_column_name)))

        self._attr_input = attr
        self._check_inputs()
        return self

    def has_defined_columns(self, role=None):
        """
        Tells if operator ``<<`` or parameter *columns*
        was used.
        """
        if role is None:
            return hasattr(self, '_attr_input') or \
                   (hasattr(self,
                            '_columns') and self._columns is not None) or \
                   (hasattr(self, 'output') and self.output is not None) \
                   or \
                   (hasattr(self, 'input') and self.input is not None)
        else:
            attr = Role.to_attribute(role)
            return hasattr(self, attr)

    def _set_role(self, role_column, role=Role.Label):
        """
        Specifies the label, the weight or the group column for a
        particular model.
        Prefix ``'w:'`` tells role is the weight,
        prefix ``'g:'`` tells role is the group,
        it is label otherwise.
        This operator is part of the syntax used
        to define a pipeline
        (see `Columns </nimbusml/concepts/columns>`_).
        """
        if isinstance(role_column, dict):
            for k, v in role_column.items():
                self._set_role(v, k)
            return self

        if isinstance(role_column, (str, tuple)):
            if role is None:
                role = Role.Label
        elif not isinstance(role_column, list):
            raise RuntimeError(
                "Unable to interpret '{0}'".format(role_column))

        if role not in DataRoles._allowed:
            # SupervisedBinner works like that.
            if not hasattr(self, 'input') or self.input is None:
                self.input = []
            if not hasattr(self, 'output') or self.output is None:
                self.output = []
            self.input.append(role_column)
            self.output.append(role)
        else:
            DataRoles.check_role(role)

            if role.endswith('column'):
                raise ValueError(
                    "role cannot end by 'column': '{0}'".format(role))
            if role is None:
                raise ValueError(
                    "role must be defined for 'column': '{0}'".format(
                        role_column))

            # Special handling for Role.GroupId.
            # This is because role GroupId maps to low level attribute
            # 'group_id_column' (note '_' in group_id)
            if role == Role.GroupId:
                role = "GroupId"

            if not self._use_role(role):
                raise RuntimeError(
                    "This learner or transform (type: '{0}') does not "
                    "use role '{1}'.".format(
                        self.type, role))
            if isinstance(role_column, (str, tuple, list)):
                attr = DataRoles.to_attribute(role)
                if ':' in role_column or ':' in attr:
                    raise ValueError(
                        "Cannot set '{0}' to '{1}'".format(
                            attr, role_column))
                self._add_attribute(attr, role_column)
            else:
                raise TypeError(
                    "role_column should be something like 'column_name' "
                    "or ('column_name', slots) not {0}".format(
                        role_column))

        return self

    @property
    def _name_or_source(self):
        """
        Some transforms need a defined output (name) but the module
        assumes it is equal to the source. Use this property instead of
        *name*
        when needed.
        """
        return self.name if hasattr(self, 'name') else self.source

    def _get_optional_column(self, colname):
        """
        Returns None if colname not found among the members
        of the class.
        """
        return getattr(self, colname) if hasattr(self, colname) else None

    def _validate_schema(self, next):
        """
        The method validates the schema of the next transform or learn
        without running it whenever possible. The schema of all
        transforms in a pipeline is usually known beforehand and this
        can be used to check the input of one layer can be found in the
        previous one.
        """
        # Not yet implemented.
        pass

    def _nodes_with_presteps(self):
        """
        The method can be used to insert preprocessing in a pipeline
        before this one.
        """
        return [self]

    def _steal_io(self, node):
        """
        This function is used when the pipeline autmatically inserts
        a step before another one. Typically, if the input column is
        integer
        and the current step requires float, the pipeline will
        automatically add
        a step to convert the input column into floats.
        The current node implements a function *Y=B(X)*
        where *X, Y* are the inputs and outputs. We need to
        insert function *A* so that we have: *Y=B(A(X))*.
        The inputs of this node are now *A(X)* instead of *X*.
        This logic is implemented by naming the new intermediate
        columns with the same output name of the original step.
        ``X >> B -> Y`` becomes ``X >> A -> Y >> B ->Y``::

            B = MinMaxScaler() << {'Y':'X'}
            A = TypeConverter()._steal_io(B)

        The method can be overloaded to make the pipeline
        automatically consider the sequence ``[A, B]``
        instead of ``[B]``.
        """
        if hasattr(node, '_columns') and node._columns is not None:
            self << node._columns

            if hasattr(node, '_attr_output'):
                setattr(node, node._attr_input,
                        getattr(node, node._attr_output))
        else:
            # No columns specified. The user plans to fit the pipeline as
            # fit(X, y).
            pass
        return self

    def _add_concatenator_node(
            self,
            data,
            input_columns,
            output_data,
            outcol,
            model):
        """
        Some transforms do not accept multiple columns but only
        one vector column and there is no ambiguity about the need
        to concatenate the column into one single column.
        This function silently adds a node into the pipeline
        without being reflected of the list of nodes the user
        explicitly defines.
        """
        from ...preprocessing.schema import ColumnConcatenator
        conc = ColumnConcatenator(columns={outcol: input_columns})
        node = conc._get_node(data=data, input=input_columns,
                              output_data=output_data, output=outcol,
                              model=model)
        node._implicit = True
        return node
