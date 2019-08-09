# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
# !/usr/bin/env python
"""
entrypoint compiler
"""

# run this script with -h or --help to get all the available options.

import argparse
import json
import os
import sys
import tempfile
from textwrap import dedent, indent, fill

from code_fixer import fix_code, fix_code_core, fix_code_entrypoint, \
    run_autopep
from codegen_checker import check_codegen
from compiler_utils import convert_name, module_to_path, \
    COPYRIGHT_NOTICE, CODEGEN_WARNING, get_presteps, tabsize
from doc_builder import DocBuilder, DocParameter
from loss_processor import write_loss, get_loss_name
from manifest_diff_parser import parse_manifest_diff_entrypoints, \
    parse_manifest_diff_components

tab = ' ' * 8


def install_and_import(package):
    import importlib
    import pip
    try:
        importlib.import_module(package)
    except ImportError:
        pip.main(
            ['install', package])
    finally:
        globals()[package] = importlib.import_module(package)


install_and_import('autoflake')
install_and_import('isort')

my_path = os.path.realpath(__file__)
my_dir = os.path.dirname(my_path)
sys.path.append(os.path.join(my_dir, '..'))
script_args = None
verbose = None


class Role:
    """
    See same class in *nimbusml*.
    """

    Feature = 'Feature'
    Label = 'Label'
    Weight = 'ExampleWeight'
    GroupId = 'RowGroup'
    # unsupported roles below
    User = 'User'
    Item = 'Item'
    Name = 'Name'
    RowId = 'RowId'

    @staticmethod
    def get_column_name(role, suffix="ColumnName"):
        """
        Converts a role into a column name
        ``GroupId --> RowGroupColumnName``.
        """
        if not isinstance(role, str):
            raise TypeError("Unexpected role '{0}'".format(role))
        if role == "Weight":
            return Role.Weight + suffix
        if role == "GroupId":
            return Role.GroupId + suffix
        return role + suffix

    @staticmethod
    def to_attribute(role, suffix="_column_name"):
        """
        Converts a role into a tuple of pythonic original and extended name.
        ``groupid --> (group_id, row_group_column_name)``.
        """
        if not isinstance(role, str):
            raise TypeError("Unexpected role '{0}'".format(role))
        if role == "weight":
            return ("weight", "example_weight" + suffix)
        if role == "groupid":
            return ("group_id", "row_group" + suffix)
        if role == "rowid":
            return ("row_id", "row_id" + suffix)
        return (role.lower(), role.lower() + suffix)

_allowed_roles = set(k for k in Role.__dict__ if k[0].upper() == k[0])

_predict_proba_template = \
    """
@trace
def predict_proba(self, X, **params):
    '''
    Returns probabilities
    '''
    return self._predict_proba(X, **params)
"""

_predict_proba_template_ova = \
    """
@trace
def predict_proba(self, X, **params):
    '''
    Returns probabilities
    '''
    self.classifier._check_implements_method('predict_proba')
    if not self.use_probabilities :
        raise ValueError('{}: use_probabilities needs to be set to True' \
        ' for predict_proba()'.format(self.__class__))
    return self._predict_proba(X, **params)
"""
_decision_function_template = \
    """
@trace
def decision_function(self, X, **params):
    '''
    Returns score values
    '''
    return self._decision_function(X, **params)
"""
_decision_function_template_ova = \
    """
@trace
def decision_function(self, X, **params):
    '''
    Returns score values
    '''
    self.classifier._check_implements_method('decision_function')
    if self.use_probabilities :
        raise ValueError('{}: use_probabilities needs to be set to False' \
         ' for decision_function()'.format(self.__class__))
    return self._decision_function(X, **params)
"""


def parse_manifest(manifest_json, pkg_path=None, overwrite=False):
    """
    """
    with open(manifest_json) as f:
        manifest = json.load(f)

    entrypoints = manifest["EntryPoints"]  # list
    nodes = []
    for e in entrypoints:
        try:
            nodes.append(
                parse_entrypoint(
                    e,
                    pkg_path=pkg_path,
                    overwrite=overwrite))
        except TypeError as exp:
            msg = '{} failed to generate. Reason: {}'.format(
                e['Name'], str(exp))
            print(msg)

    Components = manifest["Components"]  # list
    for Component in Components:
        kind = Component['Kind']
        for c in Component['Components']:
            try:
                parse_entrypoint(
                    c,
                    kind=kind,
                    pkg_path=pkg_path,
                    overwrite=overwrite)
            except TypeError as exp:
                msg = '{} failed to generate. Reason: {}'.format(
                    c['Name'], str(exp))
                print(msg)

    return [nodes]


def parse_manifest_diff(
        manifest_json,
        manifest_diff_json,
        pkg_path=None,
        overwrite=False):
    """
    """
    entrypoints = parse_manifest_diff_entrypoints(
        manifest_json, manifest_diff_json, verbose)
    entrypoint_nodes = [
        write_api(
            entrypoint,
            pkg_path=pkg_path,
            overwrite=overwrite) for entrypoint in entrypoints]
    components = parse_manifest_diff_components(
        manifest_json, manifest_diff_json, verbose)
    component_nodes = [
        write_api(
            component,
            kind=component['Kind'],
            pkg_path=pkg_path,
            overwrite=overwrite) for component in components]
    return [entrypoint_nodes] + [component_nodes]


def write_api(entrypoint, kind="node", pkg_path=None, overwrite=False):
    """
    """
    entrypoint_name = entrypoint['Name'].replace(".", "_").lower()
    class_name = entrypoint['NewName']
    class_dir = entrypoint['Module']
    class_type = entrypoint['Type']
    class_file = '_' + class_name.lower()

    doc_builder = DocBuilder()
    doc_builder.class_name = class_name
    doc_builder.class_module = class_dir
    doc_builder.desc = entrypoint['Desc']

    doc_builder_core = DocBuilder()
    doc_builder_core.class_name = class_name
    doc_builder_core.class_module = class_dir
    doc_builder_core.desc = entrypoint['Desc']

    banner = COPYRIGHT_NOTICE + CODEGEN_WARNING

    if verbose:
        print(class_name)

    ###################################
    # create function
    funobjs = create_py(entrypoint, kind=kind)
    visible_args = [
        arg for arg in funobjs['inputs'] if isinstance(
            arg.hidden, Missing)]
    doc_args = [
        DocParameter(
            name=arg.new_name_converted,
            desc=arg.desc) for arg in visible_args if
        arg.name_converted != 'column']
    doc_builder.add_manifest_args(doc_args)
    doc_builder_core.add_manifest_args(doc_args)

    # see what column param type is
    column_arg = [
        arg for arg in visible_args if arg.name_converted == 'column']

    # columns for entrypoint
    hidden_args = [arg for arg in funobjs['inputs']
                   if not isinstance(arg.hidden, Missing)]
    columns_entrypoint = [arg for arg in hidden_args]

    # In a function header, arguments must appear in this order:
    #   * any normal arguments(name);
    #   * any default arguments (name=value);
    #   * the *name (or* in 3.X) form;
    #   * any name or name=value keyword-only arguments (in 3.X);
    #   * the **name form.
    class_args = [arg.get_arg() for arg in visible_args if isinstance(
        arg.default, Missing) and arg.name_converted != 'column']
    class_args += [arg.get_arg() for arg in visible_args if not isinstance(
        arg.default, Missing) and arg.name_converted != 'column']
    class_args = ',\n        '.join(class_args)

    entrypoint_args_map = [arg for arg in visible_args if isinstance(
        arg.default, Missing) and arg.name_converted != 'column']
    entrypoint_args_map += [arg for arg in visible_args if not isinstance(
        arg.default, Missing) and arg.name_converted != 'column']
    entrypoint_args_map = [
        "%s=%s" %
        (arg.name_converted,
         arg.name_assignment) for arg in entrypoint_args_map]
    entrypoint_args_map = '\n'.join(entrypoint_args_map)

    args_map = [arg for arg in visible_args if isinstance(
        arg.default, Missing) and arg.name_converted != 'column']
    args_map += [arg for arg in visible_args if not isinstance(
        arg.default, Missing) and arg.name_converted != 'column']

    api_args_map = [
        "%s=%s" %
        (arg.new_name_converted,
         arg.new_name_converted) for arg in args_map]
    api_args_map = '\n'.join(api_args_map)

    core_args_map = [
        "%s=%s" %
        (arg.new_name_converted,
         arg.name_core_assignment) for arg in args_map]
    core_args_map = '\n'.join(core_args_map)

    fun_settings_body = None
    if class_type == 'Component':
        fun_settings_body = "\n    ".join(
            [arg.get_body() for arg in funobjs['settings']])

    dots = "..."
    if "." in class_dir:
        dots = "...."
    imports = [
        arg.get_import(
            prefix=(
                    "%sentrypoints." %
                    dots)) for arg in visible_args if
        arg.get_import() is not None]
    imports = '\n'.join(imports)

    # write the class to a file
    py_path = module_to_path(class_dir, pkg_path)
    if not os.path.exists(py_path):
        os.makedirs(py_path)
    file = os.path.join(py_path, ".".join([class_file, "py"]))
    if os.path.exists(file) and not overwrite:
        raise FileExistsError(
            "file {} exists, set 'overwrite = TRUE' to overwrite.".format(
                file))

    write_class(
        entrypoint,
        class_name,
        class_type,
        file,
        class_file,
        class_dir,
        banner,
        class_args,
        api_args_map,
        doc_builder,
        column_arg,
        hidden_args)

    # Generating test classes is broken. Commented out for now.
    # write the class test to a file
    # py_path = os.path.join(pkg_path, "tests", *class_dir.split("."))
    # if not os.path.exists(py_path): os.makedirs(py_path)
    #
    # file = os.path.join(py_path, "test_" + ".".join([class_file, "py"]))
    # if os.path.exists(file) and not overwrite:
    #     raise FileExistsError("file {} exists, set 'overwrite = TRUE'
    # to overwrite.".format(file))
    #
    # write_class_test(class_name, file)

    # write the core class to a file
    py_path = os.path.join(pkg_path, "internal", "core",
                           *class_dir.split("."))
    if not os.path.exists(py_path):
        os.makedirs(py_path)
    file = os.path.join(py_path, ".".join([class_file, "py"]))
    if os.path.exists(file) and not overwrite:
        raise FileExistsError(
            "file {} exists, set 'overwrite = TRUE' to overwrite.".format(
                file))

    write_core_class(
        entrypoint,
        entrypoint_name,
        class_name,
        class_type,
        file,
        class_file,
        class_dir,
        banner,
        imports,
        class_args,
        core_args_map,
        entrypoint_args_map,
        doc_builder_core,
        column_arg,
        columns_entrypoint,
        fun_settings_body,
        hidden_args)

    return funobjs


def parse_entrypoint(
        entrypoint,
        kind="node",
        prefix=None,
        pkg_path=None,
        overwrite=False):
    """
    """
    name = entrypoint['Name']
    # NOTE: '.' is not allowed in function name.
    fun_name = name.replace(".", "_")
    fun_file = fun_name
    # NOTE: components have to be imported before entrypoints,
    #   so use '_' prefix to differentiate components from entrypoints.
    if kind != "node":
        fun_file = "_{}_{}".format(kind, fun_file)
        fun_name = convert_name(fun_name)
    else:
        fun_name = fun_name.lower()

    if prefix is not None:
        fun_name = prefix + fun_name
    banner = CODEGEN_WARNING
    if verbose:
        print(fun_name)

    ###################################
    # create function
    funobjs = create_py(entrypoint, kind=kind)
    if entrypoint['Desc'] is not None:
        entrypoint['Desc'] = \
            fill(entrypoint['Desc'], 69, subsequent_indent=tab)
    doc_desc = "    **Description**\n        {}\n".format(
        entrypoint['Desc'])
    doc_args = [
        "    :param {name_converted}: {desc} ({inout}).".format(
            name_converted=setting.name_converted,
            desc=setting.desc,
            inout=setting.inout) for setting in funobjs['settings']]
    doc_args = [fill(item, 69, subsequent_indent=tab) for item in doc_args]
    fun_doc = "\n".join([doc_desc, *doc_args])
    fun_doc = '    """\n{}\n    """\n\n'.format(fun_doc)

    # In a function header, arguments must appear in this order:
    #   * any normal arguments(name);
    #   * any default arguments (name=value);
    #   * the *name (or* in 3.X) form;
    #   * any name or name=value keyword-only arguments (in 3.X);
    #   * the **name form.
    fun_args = [arg.get_arg() for arg in funobjs['settings']
                if isinstance(arg.default, Missing)]
    fun_args += [arg.get_arg() for arg in funobjs['settings']
                 if not isinstance(arg.default, Missing)]
    fun_args.append('**params')
    fun_args = ',\n        '.join(fun_args)
    # fun_args = ', '.join([arg.get_arg() for arg in funobjs['settings']])

    fun_imports = [arg.get_import() for arg in funobjs['settings']
                   if arg.get_import() is not None]
    fun_imports = '\n'.join(fun_imports)

    # fun_header = 'def {}(\n        {}) -> "{}":\n\n'.format(fun_name,
    # fun_args, kind)
    fun_header = 'def {}(\n        {}):\n\n'.format(fun_name, fun_args)
    fun_body = "\n    ".join(
        [arg.get_body() for arg in funobjs['settings']])

    # write the function to a file
    py_path = os.path.join(pkg_path, "internal", "entrypoints")
    if not os.path.exists(py_path):
        os.makedirs(py_path)
    file = os.path.join(py_path, ".".join([fun_file.lower(), "py"]))
    if os.path.exists(file) and not overwrite:
        raise FileExistsError(
            "file {} exists, set 'overwrite = TRUE' to overwrite.".format(
                file))

    if kind == "node":
        write_entrypoint(
            name,
            kind,
            file,
            banner,
            fun_imports,
            fun_header,
            fun_body,
            fun_doc)
    else:
        write_component(
            name,
            kind,
            file,
            banner,
            fun_imports,
            fun_header,
            fun_body,
            fun_doc)
        # write_component_import(name, fun_file)

    return funobjs


def check_read_file(name, to_write, do_check=True):
    if os.path.exists(name):
        with open(name, 'r') as f:
            content = f.read()
    else:
        content = ""
    if do_check:
        if "source=input" in to_write:
            raise Exception("name: {0}\n{1}".format(name, to_write))
        if "input=input" in to_write:
            raise Exception("name: {0}\n{1}".format(name, to_write))
        if "source: str" in to_write:
            raise Exception("name: {0}\n{1}".format(name, to_write))
    if "entrypoint" not in name:
        if ":param input:" in to_write:
            raise Exception("name: {0}\n{1}".format(name, to_write))
    return to_write == content


def write_entrypoint(
        name,
        kind,
        file,
        banner,
        fun_imports,
        fun_header,
        fun_body,
        fun_doc):
    module_doc = '"""\n{}\n"""\n'.format(name)

    imports = dedent("""
        import numbers
        {}
        from ..utils.utils import try_set, unlist
        from ..utils.entrypoints import EntryPoint
        \n\n""").format(fun_imports)

    fun_tail = indent(dedent("""\n
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
        """), prefix=" " * tabsize)

    to_write = "".join([banner, module_doc, imports, fun_header, fun_doc,
                        "    entrypoint_name = '{}'\n".format(name),
                        "    inputs = {}\n    outputs = {}\n\n    ",
                        fun_body, fun_tail])

    if not check_read_file(file, to_write, do_check=False):
        with open(file, "w") as myfile:
            print("    [write_entrypoint]", os.path.abspath(file))
            myfile.write(banner)
            myfile.write(module_doc)
            myfile.write(imports)
            myfile.write(fun_header)
            myfile.write(fun_doc)
            myfile.write("    entrypoint_name = '{}'\n".format(name))
            myfile.write("    inputs = {}\n    outputs = {}\n\n    ")
            myfile.write(fun_body)
            myfile.write(fun_tail)

    fix_code_entrypoint(name, file)


def write_component(
        name,
        kind,
        file,
        banner,
        fun_imports,
        fun_header,
        fun_body,
        fun_doc):
    module_doc = '"""\n{}\n"""\n'.format(name)

    imports = dedent("""
        import numbers
        {}
        from ..utils.utils import try_set
        from ..utils.entrypoints import Component
        \n\n""").format(fun_imports)

    fun_tail = indent(dedent("""\n
        component = Component(
            name=entrypoint_name, settings=settings, kind='{}')
        return component
        """), prefix=" " * tabsize).format(kind)

    to_write = "".join([banner, module_doc, imports, fun_header, fun_doc,
                        "    entrypoint_name = '{}'\n".format(name),
                        "    settings = {}\n\n    ",
                        fun_body, fun_tail])

    if not check_read_file(file, to_write):
        with open(file, "w") as myfile:
            print("    [write_component]", os.path.abspath(file))
            myfile.write(banner)
            myfile.write(module_doc)
            myfile.write(imports)
            myfile.write(fun_header)
            myfile.write(fun_doc)
            myfile.write("    entrypoint_name = '{}'\n".format(name))
            myfile.write("    settings = {}\n\n    ")
            myfile.write(fun_body)
            myfile.write(fun_tail)

    run_autopep(file)


def write_class(
        entrypoint,
        class_name,
        class_type,
        file,
        class_file,
        class_dir,
        banner,
        class_args,
        args_map,
        doc_builder,
        column_arg,
        hidden_args):
    """
    Write classes not in folder ``internal``.
    """
    module_doc = '"""\n{}\n"""\n'.format(class_name)

    # Add parameter columns (part of the syntax) and roles.
    hidden = set(a.name for a in hidden_args)
    allowed_roles = sorted([k.lower()
                            for k in _allowed_roles if
                            Role.get_column_name(k) in hidden])
    sig_columns_roles = list(allowed_roles)

    base_file = "base_predictor"
    base_class = "BasePredictor"
    mixin_class = "ClassifierMixin"
    base_classes = "core, BasePredictor, ClassifierMixin"
    if class_type == "Transform":
        base_file = "base_transform"
        base_class = "BaseTransform"
        mixin_class = "TransformerMixin"
        base_classes = "core, BaseTransform, TransformerMixin"
        sig_columns_roles.append("columns")
    elif class_type == "Regressor":
        base_file = "base_predictor"
        base_class = "BasePredictor"
        mixin_class = "RegressorMixin"
        base_classes = "core, BasePredictor, RegressorMixin"
    elif class_type == "Clusterer":
        base_file = "base_predictor"
        base_class = "BasePredictor"
        mixin_class = "ClusterMixin"
        base_classes = "core, BasePredictor, ClusterMixin"
    elif class_type == "Component":
        base_classes = "core"

    dots = '.' * (2 + class_dir.count('.'))

    if class_type == "Component":
        import_lines = dedent("""
            __all__ = ["{}"]

            import numbers
            from {}internal.utils.utils import trace
            from {}internal.core.{}.{} import {} as core
            \n\n""").format(class_name, dots, dots, class_dir, class_file,
                            class_name)
    else:
        import_lines = dedent("""
            __all__ = ["{}"]

            import numbers
            from sklearn.base import {}
            from {}internal.utils.utils import trace
            from {}{} import {}
            from {}internal.core.{}.{} import {} as core
            \n\n""").format(class_name, mixin_class, dots, dots, base_file,
                            base_class, dots, class_dir, class_file,
                            class_name)

    # fun_imports, fun_body
    body = ''
    header = 'class {}({}):\n##CLASS_DOCSTRINGS##\n    @trace\n    ' \
             'def __init__(\n        self,'.format(class_name, base_classes)

    if column_arg:
        if column_arg[0].name_annotated == 'column: str':
            pass
        elif column_arg[0].name_annotated == 'column: list':
            arg_type = column_arg[0].type['ItemType']
            if arg_type == 'String':
                pass
            elif arg_type['Kind'] == 'Struct':
                # see if Source is among the fields
                source_field = [
                    f for f in arg_type['Fields'] if f['Name'] == 'Source']
                # see if Name is among the fields
                name_field = [
                    f for f in arg_type['Fields'] if f['Name'] == 'Source']
                if len(source_field) > 0:
                    pass
                if len(name_field) > 0:
                    pass
        elif column_arg[0].name_annotated == 'column: dict':
            pass
        else:
            print('unknown column type %s' % column_arg[0].name_annotated)

    if class_args:
        header += '\n        {},'.format(class_args)
    for h in sig_columns_roles:
        if h == 'groupid':
            h = 'group_id'
        header += '\n        {}=None,'.format(h)
    header += '\n        **params):\n'
    assert 'groupid' not in header
    assert 'colid' not in header
    assert 'rowid' not in header

    for h in sig_columns_roles:
        if h == "groupid":
            h = 'group_id'
        elif h == "rowid":
            h = 'row_id'
        elif h == "colid":
            h = 'col_id'
        if h == 'columns' and base_class == "BasePredictor":
            continue
        doc_builder._add_io_arg_desc(
            h,
            "see `Columns </nimbusml/concepts/columns>`_")

    class_docstrings = doc_builder.get_documentation()
    if 'supervised' in class_name.lower() and ':param label:' not in \
            class_docstrings:
        raise Exception(
            "Issue with '{0}'\n{1}".format(
                class_name, class_docstrings))
    if ':param input:' in class_docstrings:
        raise Exception(
            "Issue with '{0}'\n{1}".format(
                class_name, class_docstrings))
    header = header.replace('##CLASS_DOCSTRINGS##', class_docstrings)

    if args_map != '':
        for l in args_map.split('\n'):
            (var_name, var_value) = l.split('=')
            var_name = var_name.strip()
            if var_name == 'groupid':
                var_name = 'group_id'
            var_value = var_value.strip()
            body = body + '\n            {}={},'.format(var_name,
                                                        var_value)

    # input / output arguments
    body_header = ""
    body_sig_params = []
    for h in sig_columns_roles:
        # add roles as allowed parameters
        if h == "columns":
            body_header += "\n        if {0}: params['{0}'] = {0}".format(
                h)
        else:
            body_header += "\n        if '{1}' in params: raise " \
                           "NameError(\"'{1}' must be renamed to " \
                           "'{0}'\")".format(Role.to_attribute(h)[0],
                                            Role.to_attribute(h)[1])
            body_header += "\n        if {0}: params['{1}'] = {" \
                           "0}".format(Role.to_attribute(h)[0],
                                       Role.to_attribute(h)[1])
        body_sig_params.append(h)
    if 'input_columns' in header and 'columns=' in header:
        body_header += "\n        if columns: input_columns = " \
                       "sum(list(columns.values()),[]) if " \
                       "isinstance(list(columns.values())[0], list) " \
                       "else list(columns.values())"
    if 'output_columns' in header and 'columns=' in header:
        body_header += "\n        if columns: output_columns = " \
                       "list(columns.keys())"

    assert 'groupid' not in body_header
    assert 'colid' not in body_header
    assert 'rowid' not in body_header

    if class_type == "Component":
        body = "        core.__init__(\n            self,{}\n           " \
               " **params)\n".format(body)
    elif class_type == "Transform":
        clname = "BaseTransform"
        body = "{}\n        {}.__init__(self, **params)\n        " \
               "core.__init__(\n            self,{}\n            " \
               "**params)\n".format(body_header, clname, body)
    else:
        clname = "BasePredictor"
        body = "{}\n        {}.__init__(self, type='{}', **params)\n     " \
               "" \
               "   core.__init__(\n            self,{}\n            " \
               "**params)\n".format(body_header, clname,
                                    class_type.lower(), body)

    for h in body_sig_params:
        body += '        self.{0}{1}={1}\n'.format(
            '_' if h == 'columns' else '', Role.to_attribute(h)[0])

    if 'Predict_Proba' in entrypoint:
        if entrypoint['Predict_Proba'] is True:
            if class_name == 'OneVsRestClassifier':
                body += indent(_predict_proba_template_ova,
                               prefix=' ' * tabsize)
            else:
                body += indent(_predict_proba_template,
                               prefix=' ' * tabsize)

    if 'Decision_Function' in entrypoint:
        if entrypoint['Decision_Function'] is True:
            if class_name == 'OneVsRestClassifier':
                body += indent(_decision_function_template_ova,
                               prefix=' ' * tabsize)
            else:
                body += indent(_decision_function_template,
                               prefix=' ' * tabsize)

    # additional methods
    additional = """\n    def get_params(self, deep=False):\n
        \"\"\"\n        Get the parameters for this operator.
        \"\"\"\n        return core.get_params(self)\n"""

    # presteps
    additional += get_presteps(class_name)

    to_write = "".join(
        [banner, module_doc, import_lines, header, body, additional])

    assert 'groupid' not in to_write
    assert 'colid' not in to_write
    assert 'rowid' not in to_write

    if not check_read_file(file, to_write):
        with open(file, "w") as myfile:
            print("    [write_class]", os.path.abspath(file))
            myfile.write(banner)
            myfile.write(module_doc)
            myfile.write(import_lines)
            myfile.write(header)
            myfile.write(body)
            myfile.write(additional)

    fix_code(class_name, file)


def write_class_test(class_name, file):
    body = dedent("""
        import unittest
        from nimbusml.tests.estimator_checks import check_estimator
        from nimbusml import {}

        class Test{}(unittest.TestCase):

            def test_check_estimator_{}(self):
                check_estimator({})

        if __name__ == '__main__':
            unittest.main()
        \n\n""").format(class_name, class_name, class_name.lower(),
                        class_name)

    if not check_read_file(file, body):
        with open(file, "w") as myfile:
            print("    [write_class_test] write", os.path.abspath(file))
            myfile.write(body)


def write_core_class(
        entrypoint,
        entrypoint_name,
        class_name,
        class_type,
        file,
        class_file,
        class_dir,
        banner,
        imports,
        class_args,
        args_map,
        entrypoint_args_map,
        doc_builder,
        column_arg,
        columns_entrypoint,
        fun_settings_body,
        hidden_args):
    module_doc = '"""\n{}\n"""\n'.format(class_name)

    hidden = set(a.name for a in hidden_args)
    allowed_roles = sorted([k.lower()
                            for k in _allowed_roles if
                            Role.get_column_name(k) in hidden])

    dots = '.' * (1 + class_dir.count('.'))

    # Input/output extra information.
    if class_name in {'ColumnConcatenator'}:
        base_class = "BasePipelineItem"
        baseclass_sig = ", MultiOutputsSignature"
    elif class_name in {'RangeFilter'}:
        base_class = "BasePipelineItem"
        baseclass_sig = ", SingleInputAsStringSignature"
    elif class_name == 'OneHotVectorizer':
        base_class = "BasePipelineItem"
        baseclass_sig = ", EqualInputOutputSignature"
    elif class_name in {'NGramFeaturizer', 'OneHotHashVectorizer'}:
        base_class = "BasePipelineItem"
        baseclass_sig = ", SingleOutputSignature"
    elif class_name in {'Filter', 'ColumnDropper', 'TakeFilter',
                        'SkipFilter', 'RangerFilter', 'ColumnSelector'}:
        base_class = "BasePipelineItem"
        baseclass_sig = ", NoOutputSignature"
    elif class_name in {'MutualInformationSelector'}:
        base_class = "BasePipelineItem"
        baseclass_sig = ", SingleOutputSignatureWithRoles"
    elif class_name in {'MinMaxScaler'}:
        base_class = "BasePipelineItem"
        baseclass_sig = ", DefaultSignature"
    elif len(allowed_roles) > 0:
        base_class = "BasePipelineItem"
        baseclass_sig = ", DefaultSignatureWithRoles"
    else:
        base_class = "BasePipelineItem"
        baseclass_sig = ", DefaultSignature"

    if class_type == 'Component':
        base_class = "Component"

        import_lines = dedent("""
             __all__ = ["{0}"]

             import numbers
             {1}
             from {2}..utils.entrypoints import Component
             from {2}..utils.utils import trace, try_set
             \n\n""").format(class_name, imports, dots)
    else:
        import_base_class = "from {0}.base_pipeline_item import {1}{" \
                            "2}".format(dots, base_class, baseclass_sig)

        import_lines = dedent("""
            __all__ = ["{0}"]

            import numbers
            {1}
            from {2}..entrypoints.{3} import {4}
            from {2}..utils.utils import trace
            from {2}..utils.data_roles import Role
            {5}
            \n\n""").format(class_name, imports, dots, entrypoint_name,
                            entrypoint_name, import_base_class)

    # fun_imports, fun_body
    body_snip = ''
    tail_snip = ''
    tail_snip0 = ''

    if class_type == 'Component':
        header = 'class {}({}):\n##CLASS_DOCSTRINGS##\n    @trace\n    ' \
                 'def __init__(\n        self,'.format(class_name, base_class)
    else:
        header = 'class {}({}{}):\n##CLASS_DOCSTRINGS##\n    @trace\n    ' \
                 '' \
                 'def __init__(\n        self,'.format(class_name,
                                                       base_class,
                                                       baseclass_sig)

    if column_arg:
        if column_arg[0].name_annotated == 'column: str':
            # header += '\n        input: str = None,'
            # body_snip += '\n        self.input=input'
            # doc_builder.add_input_arg(is_list=False)
            tail_snip0 = indent(dedent("""
                            input_column = self.input
                            if input_column is None and 'input' in\
 all_args:
                                input_column = all_args['input']
                            if 'input' in all_args:
                                all_args.pop('input')

                            # validate input
                            if input_column is None:
                                raise ValueError("'None' input passed\
 when it cannot be none.")

                            if not isinstance(input_column, str):
                                raise ValueError("input has to be a\
 string, instead got %s" % type(
                                input_column))

                            """), prefix=" " * tabsize)
            tail_snip = '\n        column=input_column,'
        elif column_arg[0].name_annotated == 'column: list':
            arg_type = column_arg[0].type['ItemType']
            if arg_type == 'String':
                # header += '\n        input: list = None,'
                # body_snip += '\n        self.input=input'
                # doc_builder.add_input_arg(is_list=True)
                tail_snip0 = indent(dedent("""
                                input_columns = self.input
                                if input_columns is None and 'input' in\
 all_args:
                                    input_columns = all_args['input']
                                if 'input' in all_args:
                                    all_args.pop('input')

                                # validate input
                                if input_columns is None:
                                    raise ValueError("'None' input\
 passed when it cannot be none.")

                                if not isinstance(input_columns, list):
                                    raise ValueError("input has to be\
 a list of strings, instead got %s" % type(input_columns))

                                """), prefix=" " * tabsize)
                tail_snip = '\n        column=input_columns,'
            elif arg_type['Kind'] == 'Struct':
                # see if Source is among the fields
                source_field = [
                    f for f in arg_type['Fields'] if f['Name'] == 'Source']
                # see if Name is among the fields
                name_field = [
                    f for f in arg_type['Fields'] if f['Name'] == 'Name']
                if len(source_field) > 0:
                    # header += '\n        input: list = None,'
                    # body_snip += '\n        self.input=input'
                    # doc_builder.add_input_arg(is_list=True)
                    tail_snip0 = indent(dedent("""
                                    input_columns = self.input
                                    if input_columns is None and 'input'\
 in all_args:
                                        input_columns = all_args['input']
                                    if 'input' in all_args:
                                        all_args.pop('input')

                                    # validate input
                                    if input_columns is None:
                                        raise ValueError("'None' input\
 passed when it cannot be none.")

                                    if not isinstance(input_columns, list):
                                        raise ValueError("input has to\
 be a list of strings, instead got %s" % type(input_columns))

                                 """), prefix=" " * tabsize)
                    tail_snip = '\n            column=[dict(Source=col) ' \
                                'for col in input_columns] if ' \
                                'input_columns else None,'
                if len(name_field) > 0:
                    # header += '\n        output: list = None,'
                    # body_snip += '\n        self.output=output'
                    # doc_builder.add_output_arg(is_list=True)
                    tail_snip0 = indent(dedent("""
                                    input_columns = self.input
                                    if input_columns is None and 'input'\
 in all_args:
                                        input_columns = all_args['input']
                                    if 'input' in all_args:
                                        all_args.pop('input')

                                    output_columns = self.output
                                    if output_columns is None and\
 'output' in all_args:
                                        output_columns = all_args['output']
                                    if 'output' in all_args:
                                        all_args.pop('output')
                                    __CONC__
                                    # validate input
                                    if input_columns is None:
                                        raise ValueError("'None' input\
 passed when it cannot be none.")

                                    if not isinstance(input_columns, list):
                                        raise ValueError("input has to\
 be a list of strings, instead got %s" % type(input_columns))

                                    # validate output
                                    if output_columns is None:
                                        output_columns = input_columns

                                    if not isinstance(output_columns,\
 list):
                                        raise ValueError("output has to\
 be a list of strings, instead got %s" % type(output_columns))

                                 """), prefix=" " * tabsize)
                    if entrypoint_name == "transforms_pcacalculator":
                        rep = indent(dedent("""
                                            if any(isinstance(el, list)\
 for el in input_columns):
                                                input_columns = [y for x\
 in input_columns for y in x]
                                            conc = None
                                            if isinstance(input_columns,\
 list) and len(input_columns) > 1:
                                                # We concatenate the\
 columns.
                                                data = all_args.pop("data")
                                                outcol = "temp_%s" %\
 str(id(self))
                                                if type(output_columns)\
 == list:
                                                    if len(output_columns)\
 == 1:
                                                        outcol =\
 output_columns[0]
                                                output_data = "%s_c%s" %\
 (all_args['output_data'], str(id(self)))
                                                model = "%s_c%s" %\
 (all_args['model'], str(id(self)))
                                                conc =\
 self._add_concatenator_node(
                                             data, input_columns,\
 output_data, outcol, model)
                                                input_columns = [outcol]
                                                all_args["data"] =\
 output_data
                                    """), prefix=" " * tabsize)
                        tail_snip0 = tail_snip0.replace("__CONC__", rep)
                    else:
                        tail_snip0 = tail_snip0.replace(
                            " " * tabsize + "__CONC__", "")
                    tail_snip = '\n        column=[dict(Source=i, ' \
                                'Name=o) for i, o in zip(input_columns, ' \
                                'output_columns)] if input_columns else ' \
                                'None,'
            else:
                print('unknown arg type %s' % column_arg[0].name_annotated)
        elif column_arg[0].name_annotated == 'column: dict':
            arg_type = column_arg[0].type
            if arg_type['Kind'] == 'Struct':
                # see if Source is among the fields
                source_field = [
                    f for f in arg_type['Fields'] if f['Name'] == 'Source']
                # see if Name is among the fields
                name_field = [
                    f for f in arg_type['Fields'] if f['Name'] == 'Name']
                if source_field[0]['Type']['Kind'] != 'Array' or \
                        name_field[0]['Type'] != 'String':
                    print('unknown column type')
                else:
                    tail_snip0 = indent(dedent("""
                        input_columns = self.input
                        if input_columns is None and 'input' in all_args:
                            input_columns = all_args['input']
                        if 'input' in all_args:
                            all_args.pop('input')

                        output_column = self.output
                        if output_column is None and 'output' in all_args:
                            output_column = all_args['output']
                        if 'output' in all_args:
                            all_args.pop('output')

                        # validate input
                        if input_columns is None:
                            raise ValueError("'None' input passed when\
 it cannot be none.")

                        if not isinstance(input_columns, list):
                            raise ValueError("input has to be a list of\
 strings, instead got %s"
                                             % type(input_columns))

                        # validate output
                        if output_column is None:
                            raise ValueError("'None' output passed when\
 it cannot be none.")

                        if not isinstance(output_column, str):
                            raise ValueError("output has to be a string,\
 instead got %s" % type(
                                             output_column))

                        """), prefix=" " * tabsize)
                    tail_snip = '\n        column=[dict(Source=i, ' \
                                'Name=o) for i, o in zip(input_columns, ' \
                                'output_columns)] if input_columns else ' \
                                'None,'
        else:
            print('unknown column type %s' % column_arg[0].name_annotated)

    if entrypoint_name == "schemamanipulation_concatcolumns":
        print('    [write_core_class] !=', entrypoint_name)
        tail_snip0 = indent(dedent("""
                            input_columns = self.input
                            if input_columns is None and 'input' in\
 all_args:
                                input_columns = all_args['input']
                            if 'input' in all_args:
                                all_args.pop('input')

                            output_columns = self.output
                            if output_columns is None and 'output' in\
 all_args:
                                output_columns = all_args['output']
                            if 'output' in all_args:
                                all_args.pop('output')

                            # validate input
                            if input_columns is None:
                                raise ValueError("'None' input passed\
 when it cannot be none.")

                            if not isinstance(input_columns, list):
                                raise ValueError("input has to be a list\
 of strings, instead got %s" % type(input_columns))

                            for i in input_columns:
                                if not isinstance(i, list):
                                    raise ValueError("input has to be a\
 list of list strings, instead got input element of type %s" % type(i))

                            # validate output
                            if output_columns is None:
                                raise ValueError("'None' output passed\
 when it cannot be none.")

                            if not isinstance(output_columns, list):
                                raise ValueError("output has to be a\
 list of strings, instead got %s" % type(output_columns))

                            if (len(input_columns) != len(output_columns)):
                                raise ValueError("input and output have\
 to be of same length, instead input %s and output %s" %\
 (len(input_columns), len(output_columns)))

                            column = []
                            for i in range(len(input_columns)):
                                source = []
                                for ii in input_columns[i]:
                                    source.append(ii)
                                column.append(dict([('Source', source),\
 ('Name', output_columns[i])]))
                            """), prefix=" " * tabsize)
        entrypoint_args_map = ''
        tail_snip = "column=column"

    if len(columns_entrypoint) > 0:
        for c in columns_entrypoint:
            name = c.new_name_converted
            if name.endswith('_column_name'):
                tail_snip += "\n        {0}=self._getattr_role('{0}', " \
                             "all_args),".format(name)
            elif name == "source" or c.name == "Source":
                tail_snip += "\n        source=self.source,"
            elif name == "name":
                tail_snip += "\n        name=self._name_or_source,"

    if class_args:
        header += '\n        {},'.format(class_args)
    header += '\n        **params):\n'

    assert 'groupid' not in header
    assert 'colid' not in header
    assert 'rowid' not in header
    assert 'groupid' not in tail_snip
    assert 'colid' not in tail_snip
    assert 'rowid' not in tail_snip

    class_docstrings = doc_builder.get_documentation(write_report=False)
    header = header.replace('##CLASS_DOCSTRINGS##', class_docstrings)

    body = ''
    if class_type != 'Component':
        body = "        {}.__init__(self, type='{}', **params)\n".format(
            base_class, class_type.lower())
    if body_snip != '':
        body += body_snip

    if args_map != '':
        for l in args_map.split('\n'):
            (var_name, var_value) = l.split('=')
            var_name = var_name.strip()
            var_value = var_value.strip()
            body = body + '\n        self.{}={}'.format(var_name,
                                                        var_value)

    if class_type == 'Component':
        body = body + \
               indent("\nself.kind = '{}'".format(entrypoint['Kind']),
                      prefix=" " * tabsize * 2) + \
               indent("\nself.name = '{}'".format(entrypoint['Name']),
                      prefix=" " * tabsize * 2) + \
               indent('\nself.settings={}', prefix=" " * tabsize * 2)

        fun_settings_body = fun_settings_body.replace(
            'settings[', 'self.settings[')
        body = body + '\n\n' + indent(
            '    ' + fun_settings_body, prefix=" " * tabsize)
        tail = indent(
            '\n\nsuper(' + class_name +
            ', self).__init__(name=self.name, settings=self.settings,'
            ' kind=self.kind)', prefix=" " * 2 * tabsize)
    else:
        tail = dedent("""\n
            @property
            def _entrypoint(self):
                return {0}

            @trace
            def _get_node(self, **all_args):
            """.format(entrypoint_name))

        if class_type == 'Transform':
            tail += tail_snip0

        tail += "    algo_args = dict("

        if tail_snip != '':
            tail += tail_snip

        tail = indent(tail, prefix=" " * tabsize)

        if entrypoint_args_map != '':
            for l in entrypoint_args_map.split('\n'):
                (var_name, var_value) = l.split('=')
                var_name = var_name.strip()
                var_value = var_value.strip()
                if var_name != 'loss_function':
                    tail = tail + \
                           '\n            {}=self.{},'.format(var_name,
                                                              var_value)
                else:
                    tail = tail + \
                           '\n            {}={},'.format(var_name,
                                                         var_value)

        tail = tail.rstrip(',') + ")\n\n"
        tail = tail + "        all_args.update(algo_args)\n"
        if entrypoint_name == "transforms_pcacalculator":
            tail = tail + """
        return [conc, self._entrypoint(**all_args)] if conc\
 else self._entrypoint(**all_args)
            """
        else:
            tail = tail + "        return self._entrypoint(**all_args)"

    assert 'groupid' not in tail_snip
    assert 'groupid' not in tail
    assert 'groupid' not in body
    assert 'groupid' not in header
    assert 'groupid' not in module_doc

    custom_methods = core_custom_methods(class_name)
    to_write = "".join([banner, module_doc, import_lines,
                        header, body, tail, custom_methods])

    if len(to_write.split('expression=self.expression')) > 2:
        raise Exception(
            "-----entrypoint_args_map=\n{0}\n-----to_write=\n{1}".format(
                entrypoint_args_map, to_write))
    assert 'groupid' not in to_write
    assert 'colid' not in to_write
    assert 'rowid' not in to_write

    if not check_read_file(file, to_write):
        with open(file, "w") as myfile:
            print("    [write_core_class]", os.path.abspath(file))
            myfile.write(banner)
            myfile.write(module_doc)
            myfile.write(import_lines)
            myfile.write(header)
            myfile.write(body)
            myfile.write(tail)
            myfile.write(custom_methods)
    fix_code_core(class_name, file)


def core_custom_methods(class_name):
    #
    if class_name == "KMeansPlusPlus":
        return '''

    @trace
    def fit_predict(self, X, y=None, **params):
        """
        Fits and predicts.
        """
        self.fit(X, y=y, **params)
        return self.predict(X)
        '''
    else:
        return ""


def write_component_import(name, fun_file):
    # from .modules.linear_model.fastlinearregressor import
    # FastLinearRegressor
    file = os.path.join(pkg_path, "__init__.py")
    import_line = "from .entrypoints.{} import {}\n".format(
        fun_file.lower(), name.lower())
    with open(file, "a") as myfile:
        print("    [write_component_import]", os.path.abspath(file))
        myfile.write(import_line)


def create_py(entrypoint, kind="node"):
    # print('processing {} {}'.format(kind, entrypoint['name']))
    # name = entrypoint['Name']  # str
    # desc = entrypoint['Desc']  # str
    #
    funobjs = {}
    if kind == "node":
        inputs = entrypoint['Inputs']  # list
        outputs = entrypoint['Outputs']  # list
        funobjs['inputs'] = [parse_arg(input, 'inputs') for input in
                             inputs]
        funobjs['outputs'] = [parse_arg(output, 'outputs')
                              for output in outputs]
        funobjs['settings'] = funobjs['inputs'] + funobjs['outputs']
    else:
        settings = entrypoint['Settings']  # list
        # aliases = entrypoint['aliases']     #list
        funobjs['settings'] = [
            parse_arg(
                setting,
                'settings') for setting in settings]
        funobjs['inputs'] = funobjs['settings']

    return funobjs


def parse_arg(argument, inout):
    """
    """
    if ("column" in argument['Desc'].lower() and 'column' in argument[
        "Name"].lower(
    )) or argument["Name"] in ('Name', 'Source', 'Column', "Features"):
        is_column = True
    else:
        is_column = False

    arg_type = argument['Type']
    if arg_type in ["Int", "UInt", "Float"]:
        assert not is_column
        arg_obj = NumericScalarArg(argument, inout)
    elif arg_type in ["Bool"]:
        assert not is_column
        arg_obj = BooleanScalarArg(argument, inout)
    elif arg_type in ["String", "DataView", "PredictorModel",
                      "TransformModel", "FileHandle", "Char", "Bindings"]:
        arg_obj = StringScalarArg(argument, inout, is_column=is_column)
    elif isinstance(arg_type, dict):
        kind = arg_type['Kind']
        if kind == "Enum":
            assert not is_column
            arg_obj = EnumArg(argument, inout)
        elif kind == "Struct":
            if argument['Name'] in ['Inputs', 'Outputs']:
                arg_obj = StructSubgraphArg(
                    argument, inout, is_column=is_column)
            else:
                arg_obj = StructScalarArg(argument, inout,
                                          is_column=is_column)
        elif kind == "Array":
            itemType = arg_type['ItemType']
            if itemType in ["Int"]:
                assert not is_column
                arg_obj = NumericArrayArg(argument, inout)
            elif itemType in ["String", "DataView", "PredictorModel",
                              "TransformModel", "Node"]:
                arg_obj = StringArrayArg(argument, inout,
                                         is_column=is_column)
            elif isinstance(itemType, dict):
                arg_obj = StructArrayArg(argument, inout,
                                         is_column=is_column)
            else:
                raise TypeError(
                    "unknow itemType ({}) of argument kind ({}).".format(
                        itemType, kind))
        elif kind == "Dictionary":
            itemType = arg_type['itemType']
            if itemType == "String":
                arg_obj = StringDictionaryArg(argument, inout)
            else:
                raise TypeError(
                    "unknow itemType ({}) of argument kind ({}).".format(
                        itemType, kind))
        elif kind == "Component":
            componentKind = arg_type['ComponentKind']
            if componentKind in ["NetDefinition",
                                 "Optimizer",
                                 "EarlyStoppingCriterion",
                                 "MathPlatformKind",
                                 "MultiClassClassifierNetDefinition",
                                 "CalibratorTrainer",
                                 "ParallelTraining",
                                 "AnomalyKernel",
                                 "StopWordsRemover",
                                 "NgramExtractor",
                                 "CountTableBuilder",
                                 "DistributedTrainer",
                                 "BoosterParameterFunction",
                                 "ParallelLightGBM",
                                 "AutoMlEngine",
                                 "SearchTerminator",
                                 "EnsembleSubsetSelector",
                                 "EnsembleFeatureSelector",
                                 "EnsembleMulticlassSubModelSelector",
                                 "EnsembleMulticlassDiversityMeasure",
                                 "EnsembleMulticlassOutputCombiner",
                                 "EnsembleRegressionSubModelSelector",
                                 "EnsembleRegressionDiversityMeasure",
                                 "EnsembleRegressionOutputCombiner"]:
                arg_obj = ComponentArg(argument, inout)
            elif componentKind in ["ClassificationLossFunction",
                                   "RegressionLossFunction",
                                   "SDCAClassificationLossFunction",
                                   "SDCARegressionLossFunction",
                                   "LossFunction"]:
                arg_obj = LossComponentArg(componentKind, argument, inout)
            else:
                raise TypeError(
                    "unknown componentKind ({}) of argument kind ({"
                    "}).".format(componentKind, kind))
        else:
            raise TypeError(
                "unknown kind ({}) of argument type ({}).".format(
                    kind, arg_type))
    else:
        raise TypeError("unknown argument type ({}).".format(arg_type))
    #
    return arg_obj


class Missing():
    """
    represent a missing object
    """

    def __str__(self):
        return ""


def quote(obj):
    """
    convert to a string
    """
    if isinstance(obj, str):
        res = "'{}'".format(obj)
    else:
        res = str(obj)
    return res


class Argument:
    """
    argument
    """

    def __init__(self, argument, inout):  # dict
        if not isinstance(argument, dict):
            raise ValueError("argument should be of type 'dict'.")
        self.inout = inout
        self.name = argument.get('Name', Missing())
        self.new_name = argument.get('NewName', Missing())
        self.hidden = argument.get('Hidden', Missing())
        self.type = argument.get('Type', Missing())
        self.desc = argument.get('Desc', Missing())
        self.range = argument.get('Range', Missing())
        self.default = argument.get('Default', Missing())
        self.required = argument.get('Required', Missing())
        self.aliases = argument.get('Aliases', Missing())
        self.pass_as = argument.get('PassAs', None)

        self.name_converted = convert_name(self.name)
        self.new_name_converted = convert_name(
            self.name) if isinstance(
            self.new_name,
            Missing) else convert_name(
            self.new_name)
        self.name_assignment = self.new_name_converted
        self.name_core_assignment = self.new_name_converted
        self.name_annotated = '{}: {}'.format(
            self.new_name_converted, self.type_python)

    def __str__(self):
        return self.name

    def get_import(self):
        """
        get the import statement portion for the argument
        """
        pass

    @property
    def type_python(self):
        return


class NumericScalarArg(Argument):
    """
    argument of numeric scalar type
    """

    def get_arg(self):
        """
        get the function header portion for the argument
        """
        if isinstance(self.default, Missing):
            if self.required is True:
                arg = ' = '.join([self.new_name_converted, '0'])
            else:
                arg = ' = '.join([self.new_name_converted, str(None)])
        else:
            arg = ' = '.join([self.new_name_converted, str(self.default)])
        return arg

    def get_body(self):
        """
        get the function body portion for the argument
        """
        template = "if {name_converted} is not None:\n        {inout}['{" \
                   "name}'] = try_set(\nobj={name_converted},\n" \
                   "none_acceptable={none_acceptable},\n" \
                   "is_of_type=numbers.Real"
        body = template.format(
            inout=self.inout,
            name=self.pass_as or self.name,
            name_converted=self.name_converted,
            none_acceptable=not self.required)
        if not isinstance(self.range, Missing):
            sorted_range = json.dumps(
                self.range, sort_keys=True).replace(
                '"', "'")
            range_check = ", valid_range={0}".format(sorted_range)
        else:
            range_check = ""
        return body + range_check + ")"

    @property
    def type_python(self):
        return "numbers.Real"


class BooleanScalarArg(NumericScalarArg):
    """
    argument of boolean scalar type
    """

    def get_body(self):
        """
        get the function body portion for the argument
        """
        template = "if {name_converted} is not None:\n        {inout}['{" \
                   "name}'] = try_set(obj={name_converted}, " \
                   "none_acceptable={none_acceptable}, is_of_type=bool"
        body = template.format(
            inout=self.inout,
            name=self.pass_as or self.name,
            name_converted=self.name_converted,
            none_acceptable=not self.required)
        return body + ")"

    @property
    def type_python(self):
        return "bool"


class StringScalarArg(Argument):
    """
    argument of string scalar type
    """

    def __init__(self, argument, inout, is_column=False):
        Argument.__init__(self, argument, inout)
        self.is_column = is_column

    def get_arg(self):
        """
        get the function header portion for the argument
        """
        if isinstance(self.default, Missing):
            if self.required is True:
                arg = self.new_name_converted
            else:
                arg = ' = '.join([self.new_name_converted, str(None)])
        else:
            if self.default is None:
                arg = ' = '.join([self.new_name_converted, str(None)])
            else:
                arg = ' = '.join(
                    [self.new_name_converted, repr(str(self.default))])
        return arg

    def get_body(self):
        """
        get the function body portion for the argument
        """
        template = "if {name_converted} is not None:\n        {inout}['{" \
                   "name}'] = try_set(obj={name_converted}, " \
                   "none_acceptable={none_acceptable}, is_of_type=str"
        if self.is_column:
            template += ", is_column=True"
        body = template.format(
            inout=self.inout,
            name=self.pass_as or self.name,
            name_converted=self.name_converted,
            none_acceptable=not self.required)
        return body + ")"

    @property
    def type_python(self):
        return "str"


class EnumArg(StringScalarArg):  # kind = 'Enum', values = []
    """
    argument of enum type
    """

    def get_body(self):
        """
        get the function body portion for the argument
        """
        template = "if {name_converted} is not None:\n        {inout}['{" \
                   "name}'] = try_set(obj={name_converted}, " \
                   "none_acceptable={none_acceptable}, is_of_type=str"
        body = template.format(
            inout=self.inout,
            name=self.pass_as or self.name,
            name_converted=self.name_converted,
            none_acceptable=not self.required)
        value_check = ", values={0}".format(str(self.type['Values']))
        return body + value_check + ")"


class ArrayArg(Argument):
    """
    argument of array type
    """

    def get_arg(self):
        """
        get the function header portion for the argument
        """
        if isinstance(self.default, Missing):
            arg = self.new_name_converted
        else:
            arg = ' = '.join([self.new_name_converted, str(self.default)])
        return arg

    def get_body(self):
        """
        get the function body portion for the argument
        """
        template = "if {name_converted} is not None:\n        {inout}['{" \
                   "name}'] = try_set(obj={name_converted}, " \
                   "none_acceptable={none_acceptable}, is_of_type=list"
        body = template.format(
            inout=self.inout,
            name=self.pass_as or self.name,
            name_converted=self.name_converted,
            none_acceptable=not self.required)
        return body + ")"

    @property
    def type_python(self):
        return "list"


class NumericArrayArg(ArrayArg):  # kind = Array, itemType = Int
    """
    argument of numeric array type
    """
    # TODO: check subtypes?
    pass


class StringArrayArg(ArrayArg):
    # kind = Array, itemType = String, DataView,
    # PredictorModel, TransformModel
    """
    argument of string array type
    """

    # TODO 2017-05-06: check subtypes?

    def __init__(self, argument, inout, is_column=False):
        ArrayArg.__init__(self, argument, inout)
        self.is_column = is_column

    def get_body(self):
        """
        get the function body portion for the argument
        """
        template = "if {name_converted} is not None:\n        {inout}['{" \
                   "name}'] = try_set(obj={name_converted}, " \
                   "none_acceptable={none_acceptable}, is_of_type=list"
        if self.is_column:
            template += ', is_column=True'
        body = template.format(
            inout=self.inout,
            name=self.pass_as or self.name,
            name_converted=self.name_converted,
            none_acceptable=not self.required)
        return body + ")"


class StructArrayArg(ArrayArg):  # kind = Array, itemType = dict
    """
    argument of struct array type
    """

    # TODO 2017-05-06: check subtypes?

    def __init__(self, argument, inout, is_column=False):
        ArrayArg.__init__(self, argument, inout)
        self.is_column = is_column

    def get_body(self):
        """
        get the function body portion for the argument
        """
        template = "if {name_converted} is not None:\n        {inout}['{" \
                   "name}'] = try_set(obj={name_converted}, " \
                   "none_acceptable={none_acceptable}, is_of_type=list"
        if self.is_column:
            template += ', is_column=True'
        body = template.format(
            inout=self.inout,
            name=self.pass_as or self.name,
            name_converted=self.name_converted,
            none_acceptable=not self.required)
        return body + ")"


class DictionaryArg(NumericScalarArg):
    """
    argument of dictionary type
    """

    def get_body(self):
        # NOTE 2017-06-20: a feature extractor has to be set to
        # None/null in order to disable it.
        # If not set, the default setting for those feature extractors
        # will get
        # used, which is not null.
        if self.name in ('CharFeatureExtractor', 'WordFeatureExtractor'):
            template = "{inout}['{name}'] = try_set(obj={" \
                       "name_converted}, none_acceptable={" \
                       "none_acceptable}, is_of_type=dict"
        else:
            template = "if {name_converted} is not None:\n        {" \
                       "inout}['{name}'] = try_set(obj={name_converted}, " \
                       "" \
                       "none_acceptable={none_acceptable}, is_of_type=dict"
        body = template.format(
            inout=self.inout,
            name=self.pass_as or self.name,
            name_converted=self.name_converted,
            none_acceptable=not self.required)
        return body + ")"

    @property
    def type_python(self):
        return "dict"


class StringDictionaryArg(DictionaryArg):
    # kind = Dictionary, itemType = String
    """
    argument of string dictionary type
    """
    # TODO: check subtypes?
    pass


class StructScalarArg(DictionaryArg):
    """
    argument of scalar struct type
    """

    def __init__(self, argument, inout, is_column=False):
        DictionaryArg.__init__(self, argument, inout)
        self.is_column = is_column

    def get_body(self):
        template = "if {name_converted} is not None:\n        {inout}['{" \
                   "name}'] = try_set(obj={name_converted}, " \
                   "none_acceptable={none_acceptable}, is_of_type=dict"
        if self.is_column:
            template += ", is_column=True"
        body = template.format(
            inout=self.inout,
            name=self.pass_as or self.name,
            name_converted=self.name_converted,
            none_acceptable=not self.required)
        field_check = ", field_names={0}".format(
            str([field['Name'] for field in self.type['Fields']]))
        return body + field_check + ")"


class StructSubgraphArg(StructScalarArg):
    """
    argument of struct type for subgraph input/output
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name_converted = self.name_converted + '_subgraph'
        self.new_name_converted = self.name_converted


class ComponentArg(DictionaryArg):
    # kind = Component, componentKind =
    # NetDefinitionComponent, Optimizer, LossFunction,
    # EarlyStoppingCriterion, MathPlatformKind,
    # MultiClassClassifierNetDefinition
    """
    argument of component type
    """

    def get_arg(self):
        if isinstance(self.default, Missing):
            arg = self.new_name_converted
        else:
            if self.default is None:
                arg = ' = '.join(
                    [self.new_name_converted, str(self.default)])
            else:
                component_name = convert_name(self.default['Name'])
                if self.default.get("Settings") is None:
                    arg = '{} = None'.format(
                        self.new_name_converted)  # , component_name)
                else:
                    settings = self.default['Settings']
                    cargs = [' = '.join([convert_name(key), quote(
                        settings[key])]) for key in settings]
                    arg = '{} = {}({})'.format(
                        self.new_name_converted, component_name,
                        ', '.join(cargs))
        return arg

    def get_import(self, prefix="."):
        if not isinstance(self.default,
                          Missing) and self.default is not None:
            component_name = convert_name(self.default['Name'])
            fun_file = "_{}_{}".format(
                self.type['ComponentKind'].lower(),
                self.default['Name'].lower())
            return "from {}{} import {}".format(
                prefix, fun_file, component_name)


class LossComponentArg(ComponentArg):
    """
    argument of loss component type
    """

    def get_factory_statement(self, statement_type):
        func = statement_type + '_loss'
        return "{}('{}', self.__class__.__name__, self.loss)".format(
            func, self.componentKind)

    def __init__(self, componentKind, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.componentKind = componentKind
        self.name_assignment = self.get_factory_statement('create')
        self.name_core_assignment = "loss; {}".format(
            self.get_factory_statement('check'))
        if (self.new_name_converted == 'loss'):
            self.is_api = True
        else:
            # This is an argument for creating entrypoint class, not api or
            # core class
            self.is_api = False

    def get_arg(self):
        if self.is_api:
            default_name = get_loss_name(self.default['Name'])
            arg = "{} = '{}'".format(self.new_name_converted, default_name)
        else:
            arg = "{} = None".format(self.name_converted)
        return arg

    def get_import(self, prefix="."):
        if self.is_api:
            prefix = prefix.replace('entrypoints', 'core')
            imp = "from {}loss.loss_factory import create_loss, " \
                  "check_loss".format(prefix)
        else:
            imp = super().get_import(prefix)
        return imp


def generate_code(pkg_path, generate_entrypoints, generate_api):
    manifest_json = os.path.join(my_dir, r'manifest.json')
    manifest_diff_json = os.path.join(my_dir, r'manifest_diff.json')
    doc_path = os.path.join(my_dir, '..', 'docs', 'docstrings')
    DocBuilder.set_doc_path(doc_path)

    if verbose:
        print("manifest_json: {}".format(os.path.realpath(manifest_json)))
        print("manifest_diff_json: {}".format(
            os.path.realpath(manifest_diff_json)))
        print("pkg_path: {}".format(os.path.realpath(pkg_path)))
        print("doc_path: {}".format(os.path.realpath(doc_path)))

    if generate_entrypoints:
        if verbose:
            print("Generating entrypoint classes...")
        parse_manifest(manifest_json, pkg_path=pkg_path, overwrite=True)

    if generate_api:
        if verbose:
            print("Generating public and core API classes...")
        write_loss(manifest_json, manifest_diff_json, pkg_path)
        parse_manifest_diff(
            manifest_json,
            manifest_diff_json,
            pkg_path=pkg_path,
            overwrite=True)


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    '--generate_api',
    action='store_true',
    help='Generates both public API and core API classes.')
arg_parser.add_argument(
    '--generate_entrypoints',
    action='store_true',
    help='Generates entrypoint classes.')
arg_parser.add_argument(
    '--check_manual_changes',
    action='store_true',
    help='Generates all classes in a temp ' +
         'location and checks that codegen and local changes match.')
arg_parser.add_argument(
    '--folder',
    default="temp",
    help='Where to generate files for manual checking ' +
         '(temp means use a temporary folder created by the OS is '
         'temporary)')

if __name__ == '__main__':

    # NOTE: manifest.json has to be checked in as core
    # build does not generate it.

    script_args = arg_parser.parse_args()
    pkg_path = os.path.join(my_dir, r'..\nimbusml')

    if script_args.check_manual_changes:
        verbose = False
        if script_args.folder == 'temp':
            codegen_dir = tempfile.mkdtemp(prefix='ep_compiler_')
        else:
            codegen_dir = os.path.abspath(script_args.folder)
            if not os.path.exists(codegen_dir):
                os.makedirs(codegen_dir)
        generate_code(codegen_dir, True, True)
        passed = check_codegen(
            pkg_path,
            codegen_dir,
            remove=script_args.folder == 'temp')
        if not passed:
            # Notify cake about failure with non-zero value.
            exit(1)
    elif script_args.generate_api or script_args.generate_entrypoints:
        verbose = True
        generate_code(
            pkg_path,
            script_args.generate_entrypoints,
            script_args.generate_api)
    else:
        print(
            'No valid option was provided. Use --help to view the '
            'options. Exiting without any operation.')
