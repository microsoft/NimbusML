# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. 
# -------------------------------------------------------------------------
import os
import distro
import subprocess
import sys
import unittest

import six
from nimbusml import __file__ as myfile


exps = [
    "Exception: 'Missing 'English.tok'",
    "Missing resource for SSWE",
    "Model file for Word Embedding transform could not "
    "be found",
    "was already trained. Its coefficients will be "
    "overwritten. Use clone() to get an untrained "
    "version of it.",
    "LdaNative.dll",
    "CacheClassesFromAssembly",
    "Your CPU supports instructions that this TensorFlow",
    "CacheClassesFromAssembly: can't map name "
    "OLSLinearRegression to Void, already mapped to Void",
    # TensorFlowScorer.py
    "tensorflow/compiler/xla/service/service.cc:168] XLA service",
    "tensorflow/compiler/xla/service/service.cc:175]   StreamExecutor device",
    "tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency:",
    "tensorflow/compiler/jit/mark_for_compilation_pass.cc:1412] (One-time warning): Not using XLA:CPU",
    # Binner.py
    "from collections import Mapping, defaultdict",
    "DeprecationWarning: Using or importing the ABCs",
    # BootStrapSample.py
    "DeprecationWarning: the imp module is deprecated",
    # PipelineWithGridSearchCV2.py
    "FutureWarning: You should specify a value for 'cv'",
    # PipelineWithGridSearchCV2.py
    "DeprecationWarning: The default of the 'iid' parameter",
    # PcaAnomalyDetector.py
    "UserWarning: Model",
    # FastLinearClassifier_iris_df.py
    "FutureWarning: elementwise comparison failed",
    # PcaAnomalyDetector_df.py
    "FutureWarning: Sorting because non-concatenation axis",
    # Image.py
    "Unable to revert mtime: /Library/Fonts",
    "Fontconfig error: Cannot load default config file",
    ]
if sys.version_info[:2] <= (3, 6):
    # This warning is new but it does not break any
    # other unit tests.
    # (3, 5) -> (3, 6) for tests on mac
    # TODO: Investigate.
    exps.append("RuntimeWarning: numpy.dtype size changed")


def get_examples():
    this = os.path.abspath(os.path.dirname(__file__))
    folder = os.path.normpath(
        os.path.join(
            this,
            '..',
            'nimbusml',
            'examples'))
    if not os.path.exists(folder):
        raise FileNotFoundError("Unable to find '{0}'.".format(folder))

    folder_files = [(folder, _) for _ in os.listdir(
        folder) if os.path.splitext(_)[-1] == '.py']
    if len(folder_files) == 0:
        raise FileNotFoundError(
            "Unable to find examples in '{0}'".format(folder))

    # also include the 'examples_from_dataframe' files
    folder_df = os.path.join(folder, 'examples_from_dataframe')
    folder_files_df = [(folder_df, _) for _ in os.listdir(
        folder_df) if os.path.splitext(_)[-1] == '.py']

    # merge details of all examples into one list
    folder_files.extend(folder_files_df)
    folder_files.sort()

    examples = []
    for folder, name in folder_files:
        if name in ['__init__.py',]:
            continue
        # skip for all linux & mac tests, windows is ok
        if os.name != "nt":
             if name in [
                'ToKeyImputer.py',
                'ToKeyImputer_df.py']:
                continue
        # skip for all linux tests, mac is ok
        if os.name == "posix" and distro.linux_distribution(full_distribution_name=False)[0] != '':
            if name in [
                # SymSgdNative fails to load on linux
                'SymSgdBinaryClassifier.py',
                'SymSgdBinaryClassifier_infert_df.py',
                # MICROSOFTML_RESOURCE_PATH needs to be setup on linux
                'CharTokenizer.py',
                'WordEmbedding.py',
                'WordEmbedding_df.py',
                'NaiveBayesClassifier_df.py']:
                continue

        examples.append((folder, name))

    return examples


class TestDocsExamples(unittest.TestCase):
    # This method is a static method of the class
    # because there were pytest fixture related
    # issues when the method was in the global scope.
    @staticmethod
    def generate_test_method(folder, name):
        def method(self):
            print("\n======== Example: %s =========== " % name)

            modpath = os.path.abspath(os.path.dirname(myfile))
            modpath = os.path.normpath(os.path.join(os.path.join(modpath), '..'))
            os.environ['PYTHONPATH'] = modpath
            os.environ['PYTHONIOENCODING'] = 'UTF-8'

            full = os.path.join(folder, name)
            python_exe = sys.executable.replace('w.exe', '.exe')
            cmd = '"{0}" -u "{1}"'.format(python_exe, full)

            if six.PY2:
                FNULL = open(os.devnull, 'w')
                p = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    stdin=FNULL,
                    shell=True)
                stdout, stderr = p.communicate()
            else:
                with subprocess.Popen(cmd,
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE,
                                      stdin=subprocess.DEVNULL,
                                      shell=True) as p:
                    stdout, stderr = p.communicate()
            stderr = stderr.decode('utf-8', errors='ignore').strip("\n\r\t ")
            stdout = stdout.decode('utf-8', errors='ignore').strip("\n\r\t ")

            errors = None
            if stderr != '':
                errors = stderr.split('\n')
                for exp in exps:
                    errors = [_ for _ in errors if exp not in _]

            if errors and (len(errors) > 1 or (len(errors) == 1 and errors[0] != '')):
                import numpy
                import pandas
                import sklearn
                versions = [
                    pandas.__version__,
                    sklearn.__version__,
                    numpy.__version__]
                print("DEBUG VERSIONS", versions)

                raise RuntimeError(
                    "Issue with\n  File '{0}'\n--CMD\n{1}\n--ERR\n{2}\n--OUT\n"
                    "{3}\n--".format(full, cmd, '\n'.join(errors), stdout))

        return method


for example in get_examples():
    test_name = 'test_%s' % example[1].replace('.py', '').lower()
    method = TestDocsExamples.generate_test_method(*example)
    setattr(TestDocsExamples, test_name, method)


if __name__ == "__main__":
    unittest.main()
