# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
import os
import re
import stat
import shutil
import argparse
import tempfile
import subprocess
from pathlib import Path
from code_fixer import run_autopep


description = """
This module helps with merging the changes from the master branch
in to the temp/docs branch. Here are the steps it takes:

1. Create a local clone of the users fork of NimbusML.

2. Create a new branch in the clone created in step (1)
   which tracks the temp/docs branch of the official
   NimbusML repository.

3. Remove all the tracked files from the local branch
   created in step (2).

4. Create a local clone of the master branch of the official
   NimbusML repository and checkout the specified commit
   (default is HEAD).

5. Copy all the tracked files from (4) in to (2).

6. Modify the files in (2) to be compatible with the
   documentation requirements.
"""


NIMBUSML_GIT_URL = 'https://github.com/microsoft/NimbusML.git'

# This list should not contain 'core/...' dirs.
# Subdirectories will not be automatically traversed
# and need to be explicitly added to this list.
ENTRYPOINT_DIRS = [
    r'src\python\nimbusml\cluster',
    r'src\python\nimbusml\decomposition',
    r'src\python\nimbusml\ensemble',
    r'src\python\nimbusml\ensemble\booster',
    r'src\python\nimbusml\ensemble\feature_selector',
    r'src\python\nimbusml\ensemble\output_combiner',
    r'src\python\nimbusml\ensemble\sub_model_selector',
    r'src\python\nimbusml\ensemble\sub_model_selector\diversity_measure',
    r'src\python\nimbusml\ensemble\subset_selector',
    r'src\python\nimbusml\feature_extraction',
    r'src\python\nimbusml\feature_extraction\categorical',
    r'src\python\nimbusml\feature_extraction\image',
    r'src\python\nimbusml\feature_extraction\text',
    r'src\python\nimbusml\feature_extraction\text\extractor',
    r'src\python\nimbusml\feature_extraction\text\stopwords',
    r'src\python\nimbusml\feature_selection',
    r'src\python\nimbusml\linear_model',
    r'src\python\nimbusml\model_selection',
    r'src\python\nimbusml\multiclass',
    r'src\python\nimbusml\naive_bayes',
    r'src\python\nimbusml\preprocessing',
    r'src\python\nimbusml\preprocessing\filter',
    r'src\python\nimbusml\preprocessing\missing_values',
    r'src\python\nimbusml\preprocessing\normalization',
    r'src\python\nimbusml\preprocessing\schema',
    r'src\python\nimbusml\preprocessing\text',
    r'src\python\nimbusml\timeseries',
]


def print_title(message):
    print('\n', '-' * 50, message, '-' * 50, sep='\n')


def get_dir_entries(directory, names_to_ignore=None, paths_to_ignore=None):
    if not names_to_ignore:
        names_to_ignore = []

    if not paths_to_ignore:
        paths_to_ignore = []

    files = {}
    sub_dirs = {}

    with os.scandir(directory) as it:
        for entry in it:
            if entry.name in names_to_ignore:
                continue

            if any([(x in entry.path) for x in paths_to_ignore]):
                continue

            if entry.is_file():
                files[entry.name] = entry

            elif entry.is_dir():
                sub_dirs[entry.name] = entry

    return files, sub_dirs


def rmdir(path):
    def remove_readonly(func, path, _):
        "Clear the readonly bit and reattempt the removal"
        os.chmod(path, stat.S_IWRITE)
        func(path)

    shutil.rmtree(path, onerror=remove_readonly)


def replace_file_contents(file_path, old, new, is_re=False):
    with open(file_path, 'rt') as f:
        contents = f.read()

    if is_re:
        contents = re.sub(old, new, contents)
    else:
        contents = contents.replace(old, new)

    with open(file_path, 'wt') as f:
        f.write(contents)


def init_target_repo(repo_dir, fork_git_url, branch_name):
    cwd = os.getcwd()

    if os.path.isdir(repo_dir):
        print(f'Directory {repo_dir} already exists. Removing it...')
        rmdir(repo_dir)

    print_title(f'Cloning repository {fork_git_url} in to {repo_dir}...')
    os.mkdir(repo_dir)
    os.chdir(repo_dir)
    subprocess.run(['git', 'clone', fork_git_url, '.'])
    subprocess.run(['git', 'remote', 'add', 'upstream', NIMBUSML_GIT_URL])

    print('\nAvailable remotes:')
    subprocess.run(['git', 'remote', '-v'])

    print_title('Fetching upstream branches and creating local branch...')
    subprocess.run(['git', 'fetch', 'upstream'])
    subprocess.run(['git', 'checkout', '-b', branch_name, '--track', 'upstream/temp/docs'])

    print('\nBranches:')
    subprocess.run(['git', 'branch', '-vv'])

    os.chdir(cwd)


def clear_repo(repo_dir):
    files, subdirs = get_dir_entries(repo_dir, names_to_ignore=['.git'])

    for dir_entry in files.values():
        os.remove(dir_entry)

    for dir_entry in subdirs.values():
        rmdir(dir_entry)


def git_add_all_modifications(repo_dir):
    cwd = os.getcwd()
    os.chdir(repo_dir)
    subprocess.run(['git', 'add', '-A'])
    os.chdir(cwd)


def get_master_repo(commit=None):
    tmp_dir = tempfile.mkdtemp()
    cwd = os.getcwd()
    os.chdir(tmp_dir)

    commit_name = commit if commit else 'HEAD'
    print_title(f'Cloning master branch from {NIMBUSML_GIT_URL} in to {tmp_dir} at commit {commit_name}...')
    subprocess.run(['git', 'clone', NIMBUSML_GIT_URL, '.'])

    if commit:
        subprocess.run(['git', 'checkout', commit])

    os.chdir(cwd)
    return tmp_dir


def copy_to_dir(dst, src_files, src_dirs):
    for dir_entry in src_files.values():
        shutil.copy2(dir_entry, dst)

    for dir_entry in src_dirs.values():
        shutil.copytree(dir_entry, os.path.join(dst, dir_entry.name))


def update_entrypoint_compiler(repo_dir):
    print_title('Updating entrypoint_compiler...')

    path = os.path.join(repo_dir, 'src', 'python', 'tools', 'entrypoint_compiler.py')
    replace_file_contents(path,
                          'class_file = class_name.lower()',
                          "class_file = '_' + class_name.lower()")

    print('entrypoint_compiler.py updated.')


def rename_data_dir(repo_dir):
    print_title('Renaming data directory...')

    datasets_dir = os.path.join(repo_dir, 'src', 'python', 'nimbusml', 'datasets')
    data_dir_src = os.path.join(datasets_dir, 'data')
    data_dir_dst = os.path.join(datasets_dir, '_data')
    os.rename(data_dir_src, data_dir_dst)

    path = os.path.join(repo_dir, 'src', 'python', 'nimbusml.pyproj')
    replace_file_contents(path, 'nimbusml\\datasets\\data\\', 'nimbusml\\datasets\\_data\\')

    # Update the dataset.py file to fix the data dir references
    replace_file_contents(os.path.join(datasets_dir, 'datasets.py'),
                          r'([\r\n]DATA_DIR.+)data',
                          r'\g<1>_data',
                          True)

    print('Data directory renamed.')


def rename_entrypoint_file(dir_entry):
    module_name = dir_entry.name.replace('.py', '')
    print(f'Renaming module: {module_name}\n\t({dir_entry.path})\n')

    # Update the import statement in the public file
    replace_file_contents(dir_entry.path,
                          r'(?s)([\r\n]from\s+.*\.){0}'.format(module_name),
                          r'\g<1>_{0}'.format(module_name),
                          True)

    # Rename the public file to have an underscore as its first character
    new_path = os.path.join(os.path.dirname(dir_entry), f'_{dir_entry.name}')
    os.rename(dir_entry.path, new_path)

    # Run autopep on the modified file since the modifications
    # might require new formatting which entrypoint_compiler is
    # expecting when run with the --check_manual_changes option.
    if not new_path.endswith('_cv.py'):
        run_autopep(new_path)

    # Update the import statement in __init__.py
    init_path = os.path.join(os.path.dirname(dir_entry), '__init__.py')
    replace_file_contents(init_path,
                          r'(^from\s+.*\.|[\r\n]from\s+.*\.){0}'.format(module_name),
                          r'\g<1>_{0}'.format(module_name),
                          True)

    parts = Path(dir_entry).parts
    last_index = max(i for i, val in enumerate(parts) if val == 'nimbusml')

    base_dir = os.path.join(*parts[:last_index])
    package_dir = os.path.join(*parts[last_index:-1])
    internal_dir = os.path.join(*parts[:last_index+1], 'internal', 'core', *parts[last_index+1:-1])
    internal_pkg_dir = os.path.join('nimbusml', 'internal', 'core', *parts[last_index+1:-1])

    # Rename the internal file to have an underscore as its first character
    if os.path.exists(internal_dir):
        os.rename(os.path.join(internal_dir, dir_entry.name),
                  os.path.join(internal_dir, '_' + dir_entry.name))

    # Update nimbusml.pyproj with the public and internal name changes
    replace_file_contents(os.path.join(base_dir, 'nimbusml.pyproj'),
                          os.path.join(package_dir, dir_entry.name),
                          os.path.join(package_dir, '_' + dir_entry.name))
    replace_file_contents(os.path.join(base_dir, 'nimbusml.pyproj'),
                          os.path.join(internal_pkg_dir, dir_entry.name),
                          os.path.join(internal_pkg_dir, '_' + dir_entry.name))


def rename_entrypoints(repo_dir):
    print_title('Renaming entry point files...')

    for ep_dir in ENTRYPOINT_DIRS:
        path = os.path.join(repo_dir, ep_dir)
        files, _ = get_dir_entries(path)

        for dir_entry in files.values():
            if dir_entry.name.endswith('.py') and not dir_entry.name == '__init__.py':
                rename_entrypoint_file(dir_entry)


def rename_pipeline(repo_dir):
    nimbusml_path = os.path.join(repo_dir, 'src', 'python', 'nimbusml')
    os.rename(os.path.join(nimbusml_path, 'pipeline.py'),
              os.path.join(nimbusml_path, '_pipeline.py'))

    replace_file_contents(os.path.join(nimbusml_path, '__init__.py'),
                          'from .pipeline import Pipeline',
                          'from ._pipeline import Pipeline')

    replace_file_contents(os.path.join(nimbusml_path, '__init__.py.in'),
                          'from .pipeline import Pipeline',
                          'from ._pipeline import Pipeline')

    replace_file_contents(os.path.join(repo_dir, 'src', 'python', 'nimbusml.pyproj'),
                          r'nimbusml\pipeline.py',
                          r'nimbusml\_pipeline.py')

    replace_file_contents(os.path.join(nimbusml_path, 'tests', 'test_syntax_expected_failures.py'),
                          'from nimbusml.pipeline import TrainedWarning',
                          'from nimbusml._pipeline import TrainedWarning')


# TODO: the fixes in this method shouldn't be necessary.
def fix_files(repo_dir):
    stopwords_dir = os.path.join(repo_dir,
                                 'src', 'python', 'nimbusml',
                                 'feature_extraction', 'text',
                                 'stopwords')

    replace_file_contents(os.path.join(stopwords_dir, '_customstopwordsremover.py'),
                          '__all__ = ["CustomStopWordsRemover"]',
                          '__all__ = ["CustomStopWordsRemover"]\n')

    replace_file_contents(os.path.join(stopwords_dir, '_predefinedstopwordsremover.py'),
                          '__all__ = ["PredefinedStopWordsRemover"]',
                          '__all__ = ["PredefinedStopWordsRemover"]\n')


def parse_command_line():
    global description
    arg_parser = argparse.ArgumentParser(description=description)

    arg_parser.add_argument('repo_dir',
                            help='The location on disk where to create the new local '
                            'repo which will contain the updated temp/docs branch.',
                            type=str)

    arg_parser.add_argument('fork_git_url',
                            help='The url to use for the local repository. This will usually '
                            'be the users forked repository.',
                            type=str)

    arg_parser.add_argument('branch_name',
                            help='The name of the new branch which will track temp/docs. '
                            'This branch will be created in the locally cloned copy of the '
                            'repo pointed to by fork_git_url.',
                            type=str)

    arg_parser.add_argument('-c', '--commit', help='The latest commit to include in the changes '
                            'for the new local temp/docs branch.',
                            type=str)

    args = arg_parser.parse_args()
    return args


def main():
    args = parse_command_line()

    repo_dir = Path(args.repo_dir).resolve()

    init_target_repo(repo_dir,
                     args.fork_git_url,
                     args.branch_name)

    clear_repo(repo_dir)

    master_repo_dir = get_master_repo(args.commit)

    entries = get_dir_entries(master_repo_dir, names_to_ignore=['.git'])
    copy_to_dir(repo_dir, *entries)

    rmdir(master_repo_dir)

    update_entrypoint_compiler(repo_dir)
    rename_data_dir(repo_dir)
    rename_entrypoints(repo_dir)
    rename_pipeline(repo_dir)

    fix_files(repo_dir)

    git_add_all_modifications(repo_dir)


if __name__ == '__main__':
    main()
