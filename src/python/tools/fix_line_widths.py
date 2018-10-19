# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
import os
import textwrap
from os.path import isfile
from os.path import join as pjoin

dir_path = r'src\python\docs\docstrings'
files = []
for root, directories, filenames in os.walk(dir_path):
    for file in filenames:
        f = pjoin(root, file)
        if isfile(f) and f.endswith('.txt') and ('__init__' not in f):
            files.append(f)

wrapper = textwrap.TextWrapper()
wrapper.replace_whitespace = False
wrapper.expand_tabs = False
wrapper.break_long_words = False
loss_indent = '\t'
loss_width = 69
default_indent = '        '
default_width = 77

indent = default_indent
width = default_width

wrapper.subsequent_indent = indent
wrapper.width = width
# wrapper.fix_sentence_endings=True
for filename in files:
    with open(filename, 'r+') as f:
        lines = f.readlines()
        newlines = []
        for l in lines:
            if len(l) < width:
                newlines.append(l)
            elif l.startswith(indent + ':py:class:'):
                newlines.append(l.replace(', ', '\n' + indent))
            else:
                newlines.append(wrapper.fill(l))
                newlines.append('\n')
        f.seek(0)
        f.writelines(newlines)
        f.truncate()
