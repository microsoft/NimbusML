# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
"""
Markdown tables in RST files are not handled correctly. Post-process the
generated md file to fix processing
"""
import os
import re


def createlinktoyaml(m):
    cname = m.groups()[0]
    cpath = m.groups()[1]
    cpath = cpath.replace('loss.md#', 'loss/')
    cpath = cpath.split('#')[0]
    cpath = cpath.replace('.md', '')
    cpath = cpath.replace('-', '_')
    cpath = cpath.replace('modules', 'nimbusml')
    paths = cpath.split('/')
    paths[-1] = cname
    link = '.'.join(paths)
    return '[`{}`]({})'.format(cname, "xref:" + link)


print('[fix_apiguide] Starting')
fname = os.path.join(
    os.path.normpath(__file__),
    '..',
    '..',
    '_build',
    'md',
    'apiguide.md')

with open(fname, 'r') as infile:
    lines = infile.readlines()

with open(fname, 'w') as outfile:
    for line in lines:
        line = line.replace(',,,', '---')
        line = line.replace(',,', '|')
        p = re.compile(r'\[`(.*?)`\]\((.*?)\)')
        line = p.sub(createlinktoyaml, line)
        outfile.write(line)

print('[fix_apiguide] Done')


fnames = ['types.md', 'datasources.md', 'columns.md', 'roles.md', 'schema.md',
          os.path.join('..', 'loadsavemodels.md')]
for fname in fnames:
    print('[fix_highlight_{}] Starting'.format(fname))
    fname = os.path.join(
        os.path.normpath(__file__),
        '..',
        '..',
        '_build',
        'md',
        'concepts',
        fname)
    with open(fname, 'r') as infile:
        lines = infile.readlines()

    with open(fname, 'w') as outfile:
        count = 0
        for line in lines:
            if '```' in line:
                if '```py' not in line:
                    if count % 2 == 0:
                        line = line.replace('```', '```py')
                count += 1
            outfile.write(line)
    print('[fix_highlight_{}] Done'.format(fname))
