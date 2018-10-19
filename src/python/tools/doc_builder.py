# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
"""
documentation builder
"""

import os
import re
from textwrap import dedent, indent, fill

# TODO: write help
from compiler_utils import module_to_path, get_files_recursively

tabsize = 4
tab = ' ' * tabsize


def _join_lines(*lines, add_extra_newline=True):
    return '\n'.join([str(l) for l in lines]) + \
           ('\n' if add_extra_newline else '')


def _remove_extension(f):
    return os.path.splitext(f)[0]


def _consolidate_args(*args_lists):
    all_args = []
    for args in args_lists:
        all_args.extend(args)
    return all_args


doc_tags = [
    '**Description**',
    '**Details**',
    '.. seealso::',
    '.. index::',
    '.. remarks::'
    'Example']


class DocParameter():
    def __init__(self, name, desc):
        self.name = name
        self.desc = desc

    def __str__(self):
        # clean description from ending dots. We add one ourself.
        str = self.desc.lstrip().rstrip(' \n.')
        str = ":param {}: {}.".format(self.name, str)
        if '\n' not in str:
            # Desc is not multiline. Wrap it.
            str = fill(str, 75, subsequent_indent=tab)

        return str + '\n'


class DocBuilder:
    _doc_files = None

    @staticmethod
    def set_doc_path(path):
        doc_files = get_files_recursively(path)
        DocBuilder._doc_files = {
            _remove_extension(
                f.replace(
                    path,
                    '')).lower().strip('\\'): f for f in doc_files}
        DocBuilder._report_file = open(
            os.path.join(path, '_doc_report.txt'), 'w')

    def __init__(self):
        self.class_name = None
        self.class_module = None
        self.desc = None
        self._manifest_args = []
        self._io_args = []
        self._all_args = []
        self._kwargs_param = DocParameter(
            'params', 'Additional arguments sent to compute engine')

    def add_manifest_args(self, args):
        self._manifest_args.extend(args)

    def add_input_arg(self, is_list):
        # self._add_io_arg('input', is_list)
        raise Exception('Not allowed anymore.')

    def add_output_arg(self, is_list):
        # self._add_io_arg('output', is_list)
        raise Exception('Not allowed anymore.')

    def _add_io_arg(self, name, is_list):
        # template = 'List of {} columns.' if is_list else 'Name of {}
        # column.'
        # desc = template.format(name)
        # self._io_args.append(DocParameter(name, desc))
        raise Exception('Not allowed anymore.')

    def _add_io_arg_desc(self, name, desc):
        self._io_args.append(DocParameter(name, desc))

    def _check_tags(self, doc, report):
        for t in doc_tags:
            if t not in doc:
                report.append('{} tag missing.'.format(t))

    def _fix_special_case_params(self):
        if self.class_name != 'DnnFeaturizer':
            return

        # Special fix for DnnFeaturizer
        args_to_remove = [
            arg for arg in self._manifest_args if (
                    arg.name == 'source' or arg.name == 'name')]
        for arg in args_to_remove:
            self._manifest_args.remove(arg)
        # self.add_input_arg(False)
        # self.add_output_arg(False)

    def _update_params_section(self, docstring, report):
        # Find all the params defined in the docstring
        # pattern = '\:param\W(?P<name>.*)\:(?P<desc>.*)'
        param_section_ending = '\.\. note|\.\. ' \
                               'seealso|\.\. index'
        single_param_ending = '\:param|{0}'.format(param_section_ending)
        single_param_matching_pattern = '\:param\W(.*?)\:(.*?)(?=({' \
                                        '0}))'.format(single_param_ending)
        param_section_matching_pattern = '(\:param.*?)(?=({0}))'.format(
            param_section_ending)
        found_args = re.findall(
            single_param_matching_pattern,
            docstring,
            re.DOTALL)
        docstring_args = {
            name: DocParameter(
                name,
                desc) for (name, desc, _) in found_args}

        updated_args = []
        # Use docstring params description first. If no description exists,
        # fall back to manifest.
        for arg in self._all_args:
            try:
                name = arg.name
                new_arg = docstring_args[name]
                del docstring_args[name]
            except KeyError:
                new_arg = arg

            updated_args.append(new_arg)

        if docstring_args:
            report.append(
                "Docstring params that don't exist in the class "
                "arguments:")
            report.extend([str(arg) for arg in docstring_args.values()])

        updated_params_section = _join_lines(*updated_args)
        params_section = re.findall(
            param_section_matching_pattern, docstring, re.DOTALL)
        if (len(params_section) == 1):
            params_section_text = params_section[0][0]
            docstring = docstring.replace(
                params_section_text, updated_params_section)
        elif (len(params_section) == 0):
            # There are no params in docstrings. Find out where to
            # insert the
            # params descriptions.
            docstring_ending = '({0}).*$'.format(param_section_ending)
            param_section = re.finditer(
                docstring_ending, docstring, re.DOTALL | re.MULTILINE)
            try:
                param_section_index = next(param_section).start(0)
                docstring = \
                    docstring[:param_section_index] + \
                    updated_params_section + docstring[param_section_index:]
            except StopIteration:
                # Docstrings don't have any ending section. Just append
                # params
                # to the end.
                docstring = docstring + updated_params_section
        else:
            # There's something wrong with the docstring
            report.append(
                "Could find a proper params_section (found {0} matches "
                "instead of 1)".format(
                    len(params_section)))
            return docstring

        return docstring

    def _get_custom_doc(self, smart_params_backfill=True):
        if self._doc_files is None:
            return None, None

        root_level_doctring_file = self.class_name.lower()
        module_level_doctring_file = os.path.join(
            module_to_path(self.class_module), root_level_doctring_file)

        if root_level_doctring_file in self._doc_files:
            docstring_file = root_level_doctring_file
        elif module_level_doctring_file in self._doc_files:
            docstring_file = module_level_doctring_file
        else:
            # No custom documentation is available
            return None, None

        report = []

        with open(self._doc_files[docstring_file], 'r') as doc_file:
            doc = dedent(doc_file.read())
            doc = doc.replace('\t', tab)
            doc = dedent(doc)
            doc = doc.strip('\n')

        if not (doc.startswith('"""') and doc.endswith('"""')):
            report.append(
                "{}.txt doesn't have proper begin/end triple "
                "quotes".format(
                    self.class_name))

        if smart_params_backfill:
            doc = self._update_params_section(doc, report)
        self._check_tags(doc, report)

        if not report:
            report.append('Passed')

        report = _join_lines(self.class_name, *report)

        doc = indent(doc, tab) + '\n'
        return doc, report

    def _get_generic_doc(self):
        desc = _join_lines('**Description**', indent(self.desc, tab))

        doc = _join_lines('"""', desc, *self._all_args, '"""')
        doc = indent(doc, tab)
        report = [
            self.class_name,
            'Passed (No custom documentation available. Used manifest '
            'instead.)']
        report = _join_lines(*report)
        return doc, report

    def get_documentation(self, write_report=True,
                          smart_params_backfill=True):
        self._fix_special_case_params()
        # Combine io and regular arguments
        self._all_args = _consolidate_args(
            self._io_args, self._manifest_args, [
                self._kwargs_param])

        doc, report = self._get_custom_doc(smart_params_backfill)
        if doc is None:
            doc, report = self._get_generic_doc()

        if hasattr(self, '_report_file') and write_report:
            self._report_file.write(
                report + '--------------------------------\n\n')

        return doc
