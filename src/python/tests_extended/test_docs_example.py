# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
import os
import platform
import subprocess
import sys
import time
import unittest

import six
from nimbusml import __file__ as myfile


class TestDocsExamples(unittest.TestCase):

    def test_examples(self):
        this = os.path.abspath(os.path.dirname(__file__))
        fold = os.path.normpath(
            os.path.join(
                this,
                '..',
                'nimbusml',
                'examples'))
        if not os.path.exists(fold):
            raise FileNotFoundError("Unable to find '{0}'.".format(fold))

        fold_files = [(fold, _) for _ in os.listdir(
            fold) if os.path.splitext(_)[-1] == '.py']
        if len(fold_files) == 0:
            raise FileNotFoundError(
                "Unable to find examples in '{0}'".format(fold))

        # also include the 'examples_from_dataframe' files
        fold_df = os.path.join(fold, 'examples_from_dataframe')
        fold_files_df = [(fold_df, _) for _ in os.listdir(
            fold_df) if os.path.splitext(_)[-1] == '.py']

        # merge details of all examples into one list
        fold_files.extend(fold_files_df)
        fold_files.sort()

        modpath = os.path.abspath(os.path.dirname(myfile))
        modpath = os.path.normpath(os.path.join(os.path.join(modpath), '..'))
        os.environ['PYTHONPATH'] = modpath
        os.environ['PYTHONIOENCODING'] = 'UTF-8'

        ran = 0
        excs = []

        for i, (fold, name) in enumerate(fold_files):
            if name in [
                        # Bug 294481: CharTokenizer_df fails
                        # with error about variable length vector
                        'CharTokenizer_df.py',
                        # Bug todo: CustomStopWordsRemover fails on ML.NET side
                        'NGramFeaturizer2.py',
                        ]:
                continue
            # skip for all linux tests, mac is ok
            if os.name == "posix" and platform.linux_distribution()[0] != '':
                if name in [
                    # SymSgdNative fails to load on linux
                    'SymSgdBinaryClassifier.py',
                    'SymSgdBinaryClassifier_infert_df.py',
                    # MICROSOFTML_RESOURCE_PATH needs to be setup on linux
                    'CharTokenizer.py',
                    'WordEmbedding.py',
                    'WordEmbedding_df.py',
                    'NaiveBayesClassifier_df.py'
                    ]:
                    continue
            # skip for ubuntu 14 tests
            if platform.linux_distribution()[1] == 'jessie/sid':
                if name in [
                    # libdl needs to be setup
                    'Image.py',
                    'Image_df.py'
                    ]:
                    continue
            # skip for centos7 tests 
            if platform.linux_distribution()[0] == 'CentOS Linux':
                if name in [
                    # libgdiplus needs to be setup
                    'Image.py',
                    'Image_df.py'
                    ]:
                    continue

            full = os.path.join(fold, name)
            cmd = '"{0}" -u "{1}"'.format(
                sys.executable.replace(
                    'w.exe', '.exe'), full)

            begin = time.clock()
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
                with subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE,
                                      stdin=subprocess.DEVNULL,
                                      shell=True) as p:
                    stdout, stderr = p.communicate()
            total = time.clock() - begin
            stderr = stderr.decode('utf-8', errors='ignore').strip(
                "\n\r\t ")
            stdout = stdout.decode('utf-8', errors='ignore').strip(
                "\n\r\t ")
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
                "tensorflow/compiler/xla/service/service.cc:150] XLA service",
                "tensorflow/compiler/xla/service/service.cc:158]   StreamExecutor device",
                "tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency:",
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

            errors = None
            if stderr != '':
                errors = stderr.split('\n')
                for exp in exps:
                    errors = [_ for _ in errors if exp not in _]

            if errors and (len(errors) > 1 or (len(errors) == 1 and errors[0] != '')):
                excs.append(RuntimeError(
                    "Issue with\n  File '{0}'\n--CMD\n{1}\n--ERR\n{2}\n--OUT\n"
                    "{3}\n--".format(full, cmd, '\n'.join(errors), stdout)))
                print("{0}/{1} FAIL - '{2}' in {3}s".format(i + 1, len(
                        fold_files), name, total))
                if len(excs) > 1:
                    for ex in excs:
                        print('--------------')
                        print(ex)
                    raise excs[-1]
            else:
                print("{0}/{1} OK   - '{2}' in "
                      "{3}s".format(i + 1, len(fold_files), name, total))
                ran += 1

        if len(excs) > 0:
            for ex in excs[1:]:
                print('--------------')
                print(ex)
                import numpy
                import pandas
                import sklearn
                versions = [
                    pandas.__version__,
                    sklearn.__version__,
                    numpy.__version__]
                print("DEBUG VERSIONS", versions)
            raise excs[0]
        elif ran == 0:
            raise Exception(
                "No example was run in path '{0}'.".format(fold))


if __name__ == "__main__":
    unittest.main()
