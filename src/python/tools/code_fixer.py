# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
# This file automates the manual fixes that should be made after
# entrypoint_compiler codegen
import os
import sys
import argparse
import autopep8
import autoflake
import isort

# signature_fixes contains all the signature changes that should be made after
# codegen.
# The format is class_name: [(str1, replacement1), ...] or just (str,
# replacement) for just one replacement

NG_1 = """from ...base_transform import BaseTransform"""
NG_1_correct = """from ...base_transform import BaseTransform
from .extractor import Ngram"""

FM = \
    """import numbers
from sklearn.base import ClassifierMixin
from ..internal.utils.utils import trace
from ..base_predictor import BasePredictor
from ..internal.core.decomposition.factorizationmachinebinaryclassifier \
import FactorizationMachineBinaryClassifier as core"""

FM_correct = \
    """
from sklearn.base import ClassifierMixin

from ..base_predictor import BasePredictor
from ..internal.core.decomposition.factorizationmachinebinaryclassifier \
\\\n    import FactorizationMachineBinaryClassifier as core
from ..internal.utils.utils import trace"""

OHE = \
    """import numbers
from sklearn.base import TransformerMixin
from ...internal.utils.utils import trace
from ...base_transform import BaseTransform
from ...internal.core.feature_extraction.categorical.onehothashvectorizer \
import OneHotHashVectorizer as core"""

OHE_correct = \
    """
from sklearn.base import TransformerMixin

from ...base_transform import BaseTransform
from ...internal.core.feature_extraction.categorical.onehothashvectorizer \
\\\n    import OneHotHashVectorizer as core
from ...internal.utils.utils import trace"""

cust_stop = \
    """import numbers
from ....internal.utils.utils import trace
from \
....internal.core.feature_extraction.text.stopwords.customstopwordsremover \
import CustomStopWordsRemover as core"""

cust_stop_correct = \
    """from \
\\\n....internal.core.feature_extraction.text.stopwords.customstopwordsremover\
 \\\nimport CustomStopWordsRemover as core
from ....internal.utils.utils import trace"""

pred_stop = \
    """import numbers
from ....internal.utils.utils import trace
from ....internal.core.feature_extraction.text.stopwords.predefinedstopwords\
remover import PredefinedStopWordsRemover as core"""

pred_stop_correct = \
    """from \
\\\n....internal.core.feature_extraction.text.stopwords.\
predefinedstopwordsremover \
\\\nimport PredefinedStopWordsRemover as core
from ....internal.utils.utils import trace"""

signature_fixes = {
    'SkipFilter': ('count = 0,', 'count,'),
    'TakeFilter': ('count = 9223372036854775807,', 'count,'),
    'NGramFeaturizer': [(NG_1, NG_1_correct),
                        ('word_feature_extractor = n_gram',
                         'word_feature_extractor = Ngram'),
                         ('char_feature_extractor = n_gram',
                         'char_feature_extractor = Ngram')],
    'RangeFilter': ('min = None,', 'min = -1,'),
    'FactorizationMachineBinaryClassifier': (FM, FM_correct),
    'OneHotHashVectorizer': (OHE, OHE_correct),
    'CustomStopWordsRemover': (cust_stop, cust_stop_correct),
    'PredefinedStopWordsRemover': (pred_stop, pred_stop_correct)
}


def fix_code(class_name, filename):
    _fix_code(class_name, filename, signature_fixes)


columnselector_1 = """    def _get_node(self, **all_args):
        algo_args = dict(
            keep_columns=self.keep_columns,
            drop_columns=self.drop_columns,
            keep_hidden=self.keep_hidden,
            ignore_missing=self.ignore_missing)"""

columnselector_1_correct = """    def _get_node(self, **all_args):
        input_columns = self.input
        if input_columns is None and 'input' in all_args:
            input_columns = all_args['input']
        if 'input' in all_args:
            all_args.pop('input')

        # validate input
        if input_columns is None:
            raise ValueError(
                "'None' input passed when it cannot be none.")

        if not isinstance(input_columns, list):
            raise ValueError(
                "input has to be a list of strings, instead got %s" %
                type(input_columns))

        keep_columns = self.keep_columns
        if self.keep_columns is None and self.drop_columns is None:
            keep_columns = input_columns
        algo_args = dict(
            column=input_columns,
            keep_columns=keep_columns,
            drop_columns=self.drop_columns,
            keep_hidden=self.keep_hidden,
            ignore_missing=self.ignore_missing)"""

textTransform_1 = """        if not isinstance(output_column, str):
            raise ValueError("output has to be a string, instead got %s" \
% type(
                             output_column))

        algo_args = dict(
            column=[dict(Source=i, Name=o) for i, o in zip(input_columns, \
output_columns)] if input_columns else None"""

textTransform_1_correct = """        if isinstance(output_column, list) \
and len(output_column) == 1:
            output_column = output_column[0]
        if not isinstance(output_column, str):
            raise ValueError("output has to be a string, instead got %s" \
% type(output_column))

        source = []
        for i in input_columns:
            source.append(i)
        column = dict([('Source', source), ('Name', output_column)])

        algo_args = dict(
            column=column"""

concatColumns_1 = """        if not isinstance(input_columns, list):
            raise ValueError("input has to be a list of strings, instead \
got %s" % type(input_columns))

        # validate output
        if output_columns is None:
            output_columns = input_columns

        if not isinstance(output_columns, list):
            raise ValueError("output has to be a list of strings, instead \
got %s" % type(output_columns))

        algo_args = dict(
            column=[dict(Source=i, Name=o) for i, o in zip(input_columns, \
output_columns)] if input_columns else None)"""

concatColumns_1_correct = """        if not isinstance(input_columns, \
list):
            raise ValueError("input has to be a list of list of strings, \
instead got %s" % type(input_columns))

        for i in input_columns:
            if not isinstance(i, list):
                raise ValueError("input has to be a list of list strings, \
instead got input element of type %s" % type(i))

        # validate output
        if output_columns is None:
            raise ValueError("'None' output passed when it cannot be \
none.")

        if not isinstance(output_columns, list):
            raise ValueError("output has to be a list of strings, instead \
got %s" % type(output_columns))

        if (len(input_columns) != len(output_columns)):
            raise ValueError("input and output have to be of same length, \
instead input %s and output %s" % \
(len(input_columns), len(output_columns)))

        column = []
        for i in range(len(input_columns)):
            source = []
            for ii in input_columns[i]:
                source.append(ii)
            column.append(dict([('Source', source), \
('Name', output_columns[i])]))

        algo_args = dict(
                column=column
            )"""

onevsrestclassifier_1 = """        all_args.update(algo_args)"""

onevsrestclassifier_1_correct = """
        all_args.update(algo_args)
        node = self.classifier._get_node(**all_args)
        all_args['nodes'] = [node]
        all_args['output_for_sub_graph'] = {'Model' : \
all_args['predictor_model']}"""

signature_fixes_core = {
    'SkipFilter': ('count = 0,', 'count,'),
    'TakeFilter': ('count = 9223372036854775807,', 'count,'),
    'NGramFeaturizer': (textTransform_1, textTransform_1_correct),
    'ColumnConcatenator': [('output = None,', 'output = None,'),
                           (concatColumns_1, concatColumns_1_correct)],
    'ColumnSelector': [(columnselector_1, columnselector_1_correct)],
    'RangeFilter': ('min = None,', 'min = -1,'),
    'OneVsRestClassifier': [
        (onevsrestclassifier_1, onevsrestclassifier_1_correct)],
}


def fix_code_core(class_name, filename):
    _fix_code(class_name, filename, signature_fixes_core)


s_1_incorrect = """    if predictor_model is not None:
        outputs['PredictorModel'] = try_set(obj=predictor_model, \
none_acceptable=False, is_of_type=str)"""

s_1_correct = """    if model is not None:
        outputs['PredictorModel'] = try_set(obj=model, \
none_acceptable=False, is_of_type=str)"""

signature_fixes_entrypoint = {
    'Transforms.RowSkipFilter': ('count = 0,', 'count,'),
    'Transforms.RowTakeFilter': ('count = 9223372036854775807,', 'count,'),
    'Transforms.TextFeaturizer': ('column = 0,', 'column,'),
    'Transforms.ManyHeterogeneousModelCombiner': [
        ('predictor_model = None,', 'model = None,'),
        (s_1_incorrect, s_1_correct)],
    'Transforms.TwoHeterogeneousModelCombiner': [
        ('predictor_model = None,', 'model = None,'),
        (s_1_incorrect, s_1_correct)],
    'Transforms.LightLda' : ('num_threads = 0,', 'num_threads = None,'),
    'Trainers.GeneralizedAdditiveModelRegressor': ('Infinity', 'float("inf")'),
    'Trainers.GeneralizedAdditiveModelBinaryClassifier': (
        'Infinity', 'float("inf")'),
    'Models.CrossValidator': [
        ('inputs_subgraph = 0,', 'inputs_subgraph,'),
        ('outputs_subgraph = 0,', 'outputs_subgraph,')],
}


def fix_code_entrypoint(class_name, filename):
    _fix_code(class_name, filename, signature_fixes_entrypoint)


def _fix_code(class_name, filename, fixes_dict):
    if class_name in fixes_dict:
        fixes = fixes_dict[class_name]
        if not isinstance(fixes, list):
            fixes = [fixes]

        with open(filename, 'r+') as f:
            code = f.read()
            first = True
            for fix in fixes:
                code = code.replace(fix[0], fix[1])
            f.seek(0)
            f.write(code)
            f.truncate()
    run_autopep(filename)


import_fixed = \
    ["decomposition\\factorizationmachinebinaryclassifier.py",
     "feature_extraction\\categorical\\onehothashvectorizer.py",
     "feature_extraction\\text\\stopwords\\customstopwordsremover.py",
     "feature_extraction\\text\\stopwords\\predefinedstopwordsremover.py"]


def run_autopep(filename):
    cmd_args = ['dummy', '-d']
    args = autopep8.parse_args(cmd_args)
    args.aggressive = 2
    args.in_place = True
    args.diff = False
    args.max_line_length = 66
    autopep8.fix_file(filename, args)

    if not any(x in filename for x in import_fixed) or "internal" in filename:
        isort.SortImports(filename)
        run_autoflake(filename)


def run_autoflake(filename):
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-place', action='store_true')
    parser.add_argument('--remove-all-unused-imports', action='store_true')
    cmd_args = ['--in-place', '--remove-all-unused-imports']
    args = parser.parse_args(cmd_args)
    args.check = None
    args.imports = None
    args.expand_star_imports = None
    args.remove_duplicate_keys = None
    args.remove_unused_variables = None
    args.ignore_init_module_imports = False
    autoflake.fix_file(filename, args=args, standard_out=sys.stdout)
