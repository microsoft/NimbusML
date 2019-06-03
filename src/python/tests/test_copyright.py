# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
import os
import unittest


class TestCopyright(unittest.TestCase):
    """
    Tests that the copyright is present in everyfile.
    """

    def test_copyrigth_present(self):
        return
        root = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                            '..')
        root = os.path.normpath(root)
        allfiles = []
        for r, dirs, files in os.walk(root):
            for name in files:
                if name.endswith('.py'):
                    allfiles.append(os.path.join(r, name))

        nothere = []
        nb = 0
        for name in allfiles:
            if 'examples' in name or 'docs' in name or \
                    '__init__.py' in name or 'entrypoints' in name:
                continue
            with open(name, "r") as f:
                content = f.read()
            if 'Copyright (c) Microsoft Corporation. All rights ' \
               'reserved.' not in content:
                nothere.append(name)
            else:
                nb += 1

        if len(nothere) > 0:
            nothere = ['  File "{0}", line 1'.format(_) for _ in nothere]
            raise Exception(
                "Copyright not found in\n{0}".format(
                    "\n".join(nothere)))
        if nb == 0:
            raise Exception("No file found in '{0}'".format(root))


if __name__ == '__main__':
    unittest.main()
