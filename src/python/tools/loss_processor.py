# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
import os
import json
from compiler_utils import JsonArrayWithIdLookup, JsonArrayWithNameLookup, \
    update_json_object, load_json, ManifestIds, convert_name, \
    COPYRIGHT_NOTICE, CODEGEN_WARNING
from code_fixer import run_autopep
from doc_builder import DocBuilder
from collections import defaultdict
from jinja2 import Template

_api_template = \
    """{{ header }}
{%- for loss in losses %}
class {{ loss.api_class }}:
{{ loss.docstring }}
    def __init__(self {%- for name, value in loss.params %}, \
{{name}}={{value}}{% endfor -%}):
        self._params = { {%- for name, _ in loss.params -%}
            {%- if not loop.last -%}
            '{{name}}': {{name}},
            {%- else -%}
            '{{name}}': {{name}}
            {%- endif -%}
        {%- endfor -%}}
        self._string_name = '{{loss.api_str}}'
{% endfor %}
"""

_loss_table_template = \
    """{{ header }}
import json

loss_table_json = json.loads('''
{{ table }}
''')"""

_loss_dict = defaultdict()


class LossComponent:
    def __init__(self, kind, nimbusml_class, api_class, api_str, params,
                 entrypoint_module, entrypoint_function):
        self.kind = kind
        self.nimbusml_class = nimbusml_class
        self.api_class = api_class
        self.api_str = api_str
        self.params = params
        self.entrypoint_module = entrypoint_module
        self.entrypoint_function = entrypoint_function


def _extract_loss_components(manifest_json, manifest_diff_json):
    loss_list = []
    manifest = load_json(manifest_json)
    manifest_diff = load_json(manifest_diff_json)

    components = JsonArrayWithIdLookup(
        manifest["Components"],
        _id=ManifestIds.Kind.value)
    diff_components = JsonArrayWithIdLookup(
        manifest_diff["Components"],
        _id=ManifestIds.Kind.value)

    C = 'Components'
    for _id in diff_components.ids:
        if 'LossFunction' in _id:
            diff_component = diff_components[_id]
            component = components[_id]
            if C not in diff_component:
                diff_component[C] = component[C]

            diff_sub_components = JsonArrayWithNameLookup(diff_component[C])
            sub_components = JsonArrayWithNameLookup(component[C])
            for name in sub_components.names:
                sc = sub_components[name]
                diff_sc = diff_sub_components[name]
                update_json_object(sc, diff_sc)

                ep_module = "_{}_{}".format(_id.lower(), name.lower())
                ep_function = convert_name(name)

                new_name = sc.get('NewName')
                if new_name is None:
                    new_name = name.replace('Loss', '')
                string_name = convert_name(new_name)
                params = [(convert_name(s['Name']), s['Default'])
                          for s in sc['Settings']]

                doc_builder = DocBuilder()
                doc_builder.class_name = new_name
                doc_builder.class_module = 'loss'

                lc = LossComponent(
                    kind=_id,
                    nimbusml_class=name,
                    api_class=new_name,
                    api_str=string_name,
                    params=params,
                    entrypoint_module=ep_module,
                    entrypoint_function=ep_function)
                lc.docstring = doc_builder.get_documentation(
                    write_report=False, smart_params_backfill=False)
                loss_list.append(lc)
    return loss_list


def _create_loss_table(manifest_json, manifest_diff_json, pkg_path):
    loss_table = defaultdict(defaultdict)

    loss_list = _extract_loss_components(manifest_json, manifest_diff_json)

    for l in loss_list:
        _loss_dict[l.api_str] = l

        loss_table[l.kind][l.api_str] = l.entrypoint_module.strip('_')

    loss_table_file = os.path.join(
        pkg_path,
        'internal',
        'core',
        'loss',
        'loss_table_json.py')
    loss_table_fold = os.path.dirname(loss_table_file)
    if not os.path.exists(loss_table_fold):
        os.makedirs(loss_table_fold)

    # header = COPYRIGHT_NOTICE + CODEGEN_WARNING
    t = Template(_loss_table_template)
    table_text = t.render(
        table=json.dumps(
            loss_table,
            sort_keys=True,
            indent=2),
        header=COPYRIGHT_NOTICE +
        CODEGEN_WARNING)
    # loss_api_file = os.path.join(pkg_path, 'loss.py')
    if not os.path.exists(os.path.dirname(loss_table_file)):
        os.makedirs(os.path.dirname(loss_table_file))
    with open(loss_table_file, 'w') as f:
        f.write(table_text)
    run_autopep(loss_table_file)


def _write_api(pkg_path):
    losses = sorted(_loss_dict.values(), key=lambda x: x.api_class)
    code = Template(_api_template).render(
        losses=losses, header=COPYRIGHT_NOTICE + CODEGEN_WARNING)
    loss_api_file = os.path.join(pkg_path, 'loss.py')
    with open(loss_api_file, 'w') as f:
        f.write(code)
    run_autopep(loss_api_file)


def write_loss(manifest_json, manifest_diff_json, pkg_path):
    _create_loss_table(manifest_json, manifest_diff_json, pkg_path)
    _write_api(pkg_path)


def get_loss_name(nimbusml_loss):
    return convert_name(nimbusml_loss.replace('Loss', ''))
