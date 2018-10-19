# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
import unittest
import os
import re
import sys


class TestPyproj(unittest.TestCase):
    """
    Two tests to ensure that nimbusml.pyproj is in sync with
    the list of files the project contains whatever the tool used to
    edit the files is.
    """

    def test_pyproj_existence(self):
        "Tests that every file in pyproj does exist."
        if sys.platform != "win32":
            return
        this = os.path.abspath(os.path.dirname(__file__))
        pyp = os.path.normpath(
            os.path.join(
                this,
                '..',
                'nimbusml.pyproj'))
        with open(pyp, 'r') as f:
            content = f.read()

        reg = re.compile('Compile Include=.?(.+?[.]py).?"')
        names = reg.findall(content)
        assert len(names) > 10
        root = os.path.normpath(os.path.abspath(os.path.join(this, '..')))
        for name in names:
            full = os.path.join(root, name)
            if not os.path.exists(full):
                raise FileNotFoundError(
                    "root: {0}\nname: {1}\nfull: {2}".format(
                        root, name, full))

    def test_pyproj_in(self):
        """
        Tests that every python file in at the same level as nimbusml.pyproj
        is part of the pyproj file.
        """
        if sys.platform != "win32":
            return
        this = os.path.abspath(os.path.dirname(__file__))
        pyp = os.path.normpath(
            os.path.join(
                this,
                '..',
                'nimbusml.pyproj'))
        with open(pyp, 'r') as f:
            content = f.read()

        reg = re.compile('Compile Include=.?(.+?[.]py).?"')
        names = reg.findall(content)
        assert len(names) > 10
        root = os.path.normpath(os.path.join(pyp, '..'))

        allfiles = []
        for r, dirs, files in os.walk(root):
            if "build" in r:
                continue
            for name in files:
                if name.endswith('.py'):
                    if "ssweembedding" in name:
                        print(root, r, name)
                    allfiles.append(os.path.join(r, name))

        files = [os.path.relpath(_, root) for _ in allfiles]
        files = [
            _.replace(
                "nimbusml\\",
                "").replace(
                "build\\lib\\",
                "") for _ in files]
        names = [
            _.replace(
                "nimbusml\\",
                "").replace(
                "build\\lib\\",
                "") for _ in names]

        inproj = set(names)
        notfound = [
            f for f in files if f not in inproj and '__init__.py' not in f]
        if len(notfound) > 0:
            notfound = [
                '    <Compile Include="{0}" />'.format(_) for _ in notfound]
            if len(notfound) > 20:
                notfound = notfound[:20]
            raise Exception(
                "Not found in pyproj\n{0}\n---------CONTENT"
                "--------\n{1}".format('\n'.join(notfound), content))


if __name__ == '__main__':
    unittest.main()
