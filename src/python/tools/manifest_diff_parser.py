# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
"""
Parser for manifest_diff.json
"""

__all__ = ['parse_manifest_diff_entrypoints']

from compiler_utils import JsonArrayWithNameLookup, update_json_object, \
    load_json, ManifestIds

_error_workarounds = (
    '''Possible causes are the following:
    1) The name is mis-spelled in manifest_diff.json. Spelling is
    case-sensitive.
    2) If you\'ve updated manifest.json, the name might have changed.
    Please also update the name in manifest_diff.json''')
_global_input_diffs_lookup = None
_manifest_entrypoints_lookup = None


class EntryPoint:
    def __init__(self, json_obj):
        self._json_obj = json_obj
        self.name = json_obj.get(ManifestIds.Name.value)
        self.new_name = json_obj.get('NewName')
        self.inputs = JsonArrayWithNameLookup(
            json_obj.get(ManifestIds.Inputs.value, []))

    def update_properties(self, new_ep):
        # Only update simple properties (which exludes Inputs array)
        update_json_object(
            self.get_json(), new_ep.get_json(), {
                ManifestIds.Inputs.value})

    def update_inputs(self, json_array):
        common_names = self.inputs.names & json_array.names
        for n in common_names:
            update_json_object(self.inputs[n], json_array[n])

    def get_json(self):
        return self._json_obj


def _get_entrypoint_portions(ep_diff_json):
    ep_diff = EntryPoint(ep_diff_json)
    ep_name = ep_diff.name
    if ep_diff.new_name is None:
        error = 'Could not find NewName for component {} in ' \
                'manifest_diff.json.'.format(ep_name)
        raise ValueError(error)

    if ep_name not in _manifest_entrypoints_lookup:
        error = 'Could not find component {} in manifest.json. '.format(
            ep_name) + _error_workarounds
        raise ValueError(error)
    ep = EntryPoint(_manifest_entrypoints_lookup[ep_name])

    # Check all input names match
    invalid_inputs = ep_diff.inputs.names - ep.inputs.names
    if invalid_inputs:
        error = \
            'Could not find inputs {} for component {}' \
            'in manifest.json. '.format(invalid_inputs, ep_name) + \
            _error_workarounds
        raise ValueError(error)

    return ep, ep_diff


def _get_complete_entrypoint(ep_diff_json):
    ep, ep_diff = _get_entrypoint_portions(ep_diff_json)

    # Overwrite/Add all simple properties
    ep.update_properties(ep_diff)

    # Apply global input changes
    ep.update_inputs(_global_input_diffs_lookup)

    # Apply inputs changes in the diff. These will override global changes.
    ep.update_inputs(ep_diff.inputs)

    return ep.get_json()


def _report_missing_entrypoints(manifest_diff):
    manifest_all_eps = set(_manifest_entrypoints_lookup.names)
    manifest_diff_all_eps = \
        set(JsonArrayWithNameLookup(manifest_diff["EntryPoints"]).names)\
        | \
        set(manifest_diff['HiddenEntryPoints'])

    missing_eps = manifest_all_eps - manifest_diff_all_eps
    if missing_eps:
        message = '\n'.join(
            ['Warning: {} entrypoint is missing.'.format(e) for e in
             missing_eps])
        print(
            'The following entrypoints are not included in '
            'manifest_diff.json:\n',
            message)


def parse_manifest_diff_entrypoints(
        manifest_json,
        manifest_diff_json,
        report_missing):
    global _global_input_diffs_lookup, _manifest_entrypoints_lookup

    manifest = load_json(manifest_json)
    manifest_diff = load_json(manifest_diff_json)

    _manifest_entrypoints_lookup = JsonArrayWithNameLookup(
        manifest["EntryPoints"])
    _global_input_diffs_lookup = JsonArrayWithNameLookup(
        manifest_diff['GlobalChanges'][ManifestIds.Inputs.value])
    if report_missing:
        _report_missing_entrypoints(manifest_diff)
    return [_get_complete_entrypoint(ep_diff)
            for ep_diff in manifest_diff["EntryPoints"]]


def parse_manifest_diff_components(
        manifest_json,
        manifest_diff_json,
        report_missing):
    global _global_input_diffs_lookup, _manifest_entrypoints_lookup

    manifest = load_json(manifest_json)
    manifest_diff = load_json(manifest_diff_json)

    components = []
    components_diff = []

    for kind in manifest['Components']:
        for component in kind['Components']:
            component['Kind'] = kind['Kind']
            components.append(component)
    _manifest_entrypoints_lookup = JsonArrayWithNameLookup(components)

    for kind in manifest_diff['Components']:
        if 'Components' in kind:
            for component in kind['Components']:
                component['Kind'] = kind['Kind']
                components_diff.append(component)
    _global_input_diffs_lookup = JsonArrayWithNameLookup(components_diff)
    return [_get_complete_entrypoint(comp_diff)
            for comp_diff in components_diff]
