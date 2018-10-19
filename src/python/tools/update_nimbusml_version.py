# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
# !/usr/bin/env python
"""
update ML.NET version
"""

# run this file to update the ML.NET version in all the necessary files:
#   * double click it in File Explorer, or
#   * run it directly on Command Prompt, e.g., !python update_mlnet_version.py
# see the bottom section of this file for details about this updating process.

import os
import warnings


def update_version(filename, old, new):
    """ replace the version in place """
    with open(filename) as fileobj:
        file_content = fileobj.read()
        if old not in file_content:
            warnings.warn(
                "'{old}' not found in '{filename}'".format(
                    **locals()))
            return

    with open(filename, 'w') as fileobj:
        print(
            "replacing '{old}' with '{new}' in '{filename}'".format(
                **locals()))
        file_content = file_content.replace(old, new)
        fileobj.write(file_content)


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(
        description='Update version for MicrosoftML')
    parser.add_argument('old', help='the old version')
    parser.add_argument('new', help='the new version')
    args = parser.parse_args()

    old, new = args.old, args.new

    if not old or not new:
        msg = "either '{old}' or '{new}' is not a valid version number"
        raise ValueError(msg.format(**locals()))

    my_path = os.path.realpath(__file__)
    my_dir = os.path.dirname(my_path)
    code_dir = os.path.join(my_dir, '../../../code')

    files_to_update = dict(
        ManifestGenerator=["ManifestGenerator.csproj", "packages.config"],
        DotNetBridge=["DotNetBridge.csproj", "packages.config"],
        Core=["DotNetBridge/DotNetBridge.csproj", "DotNetBridge/project.json"]
    )

    paths_to_update = [
        os.path.abspath(
            os.path.join(
                code_dir,
                key,
                value)) for key,
        values in files_to_update.items() for value in values]

    for filename in paths_to_update:
        update_version(filename, old, new)

    # input("\nPress Enter to exit") # for causing a pause
