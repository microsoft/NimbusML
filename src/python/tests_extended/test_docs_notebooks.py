# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
"""
test ipython notebooks in nimbusml
"""
import json
import os
import distro
import sys
import time
import unittest

import nbconvert
import nbformat
import six


class TestDocsNotebooks(unittest.TestCase):

    # Using nbconvert to execute ipython notebooks
    # Sample example here :
    # https://github.com/elgehelge/nbrun/blob/master/nbrun.py
    @staticmethod
    def run_notebook(self, path, timeout=300, retry=3):
        print("Running notebook : {0}".format(path))

        s = time.time()
        ep = nbconvert.preprocessors.ExecutePreprocessor(
            extra_arguments=["--log-level=40"],
            timeout=timeout,
        )
        path = os.path.abspath(path)
        assert path.endswith('.ipynb')
        nb = nbformat.read(path, as_version=4)
        try:
            ep.preprocess(nb, {'metadata': {'path': os.path.dirname(path)}})
            return s
        except Exception as e:
            if isinstance(e, RuntimeError) and retry != 0:
                return self.run_notebook(
                    path, timeout=timeout, retry=retry - 1)

            raise Exception('Error running notebook {0}:  {1}'.format(path, e))

    # REVIEW: Work is needed to fix the jupyter kernel config file to run
    # without Anaconda (see comments below): Work Item 246961.
    @unittest.skipIf(
        os.name == 'nt',
        "Not supported on this platform without using Anaconda.")
    # REVIEW: Figure out how to spin off kernels on linux with no Anaconda
    @unittest.skipIf(distro.linux_distribution(full_distribution_name=False)[0] == "Ubuntu" and
                     distro.linux_distribution(full_distribution_name=False)[
                         1] == "16.04", "not supported on this platform")
    def test_notebooks(self):
        # These tests are blocking PRs b/c they fail in linux build. Disabling
        # them for now.
        if os.name != 'nt':
            return

        # need to fix jupyter kernel config file for the python interpreter we
        # use in the build machine for py3.6
        # currently the jupyter kernel config file has a hardcoded python path
        # (E:\tmp\python.exe) which is incorrect
        python2or3 = 'python2' if six.PY2 else 'python3'
        jupyter_kernel_configfile = os.path.join(
            os.path.dirname(
                sys.executable),
            'share',
            'jupyter',
            'kernels',
            python2or3,
            'kernel.json')

        if os.path.isfile(jupyter_kernel_configfile):
            with open(jupyter_kernel_configfile) as f:
                d = json.load(f)

            # set jupyter kernel config file to use the right python path
            if d['argv'][0] != sys.executable:
                d['argv'][0] = sys.executable
                with open(jupyter_kernel_configfile, 'w') as fp:
                    json.dump(d, fp)

        # get list of notebooks to run
        this = os.path.abspath(os.path.dirname(__file__))
        fold = os.path.normpath(os.path.join(this, '..', 'docs', 'notebooks'))
        tried = [fold]
        if not os.path.exists(fold):
            fold2 = os.path.join(
                fold.split('python')[0],
                'python',
                'docs',
                'notebooks')
            tried.append(fold2)
            fold = fold2
        if not os.path.exists(fold):
            raise FileNotFoundError("Unable to find '{0}'.".format(fold))

        notebooks = [_ for _ in os.listdir(
            fold) if os.path.splitext(_)[-1] == '.ipynb']
        if len(notebooks) == 0:
            raise FileNotFoundError(
                "Unable to find notebooks in '{0}'".format(fold))

        # REVIEW: remove skip list once resource download over ssl is fixed on
        # centos7 & ubuntu14 test boxes
        skip_notebooks_linux = []
        skip_notebooks_linux.append(
            '2.3 [Image] Image Processing - Clustering '
            'Using ImageNet Data.ipynb')
        skip_notebooks_linux.append(
            '2.4 [Text] Working with Scikit Learn Toolkit - Classification'
            ' Using Wikipedia Detox Data.ipynb')

        # Notebooks that don't come with datasets will be skipped
        skip_notebooks = [
            '2.3 [Image] Image Processing - Clustering Using '
            'ImageNet Data.ipynb']
        skip_notebooks.append(
            '3.5 [Numeric] Data Manipulation Using Expression.ipynb')
        skip_notebooks.append(
            '3.3 [Numeric] Performance Comparison with '
            'Flight Delay Data.ipynb')
        for i, nb in enumerate(notebooks):
            nbfullpath = os.path.join(fold, nb)
            if (os.name != 'nt' and nb in skip_notebooks_linux) \
                    or nb in skip_notebooks:
                continue
            self.run_notebook(self, nbfullpath)


if __name__ == '__main__':
    unittest.main()
