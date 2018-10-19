# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
import os
import shutil
from difflib import context_diff
from os.path import isfile
from os.path import join as pjoin


def get_python_files(dir_path):
    files = []
    for root, directories, filenames in os.walk(dir_path):
        for file in filenames:
            f = pjoin(root, file)
            if isfile(f) and f.endswith('.py') and ('__init__' not in f):
                files.append(f)
    return files


def are_identical(file1, file2, verbose=False):
    # Check if files are identical without any non-standard package
    # like filecmp or md5
    with open(file1) as f1:
        with open(file2) as f2:
            c1 = f1.readlines()
            c2 = f2.readlines()
            if c1 == c2:
                return True
            elif verbose:
                if len(c1) != len(c2):
                    print("File '{0}' - #lines={1}".format(file1, len(c1)))
                    print("File '{0}' - #lines={1}".format(file2, len(c2)))
                for i, (a, b) in enumerate(zip(c1, c2)):
                    if a != b:
                        print("First difference at line {0}".format(i))
                        print("'{0}'".format(a[:-1]))
                        print("'{0}'".format(b[:-1]))
                        break
                    else:
                        print("{0} == '{1}'".format(i, a[:-1]))
                print('\n'.join(context_diff(c1, c2,
                                             fromfile=file1,
                                             tofile=file2)))
            return False


def check_codegen(pkg_path, codegen_path, remove=True):
    errors = []
    files = get_python_files(codegen_path)
    for f in files:
        pkg_f = f.replace(codegen_path, pkg_path)
        relative_f = f.replace(codegen_path, '<nimbusml>')
        if os.path.exists(pkg_f):
            if not are_identical(f, pkg_f, True):
                msg = 'Error: File {0} content differs from codegen ' \
                      'content.'.format(relative_f)
                errors.append(msg)
        else:
            msg = 'Error: File {0} does not exist, but codegen creates ' \
                  'it.'.format(relative_f)
            errors.append(msg)

    if not errors:
        if remove:
            # Delete the temp codegen directory
            shutil.rmtree(codegen_path)
        print('Entrypoint codegen checker passed.')
        return True
    else:
        errors = '\n'.join(errors)
        print('Entrypoint codegen checker failed:')
        print(errors)
        # Keep the temp codegen directory for inspection
        print('Codegen files could be found in ', codegen_path)
        return False
