# Original Source

This code is a modified version of the code found
[here](https://github.com/robotology/gh-action-nightly-merge).

# Merge Action

Automatically merge a source branch into a target branch.

If the merge is not necessary, the action will do nothing.
If the merge fails due to conflicts, the action will fail, and the repository
maintainer should perform the merge manually.

## Installation

To enable the action, create the `.github/workflows/nightly-merge.yml`
file with the following content:

```yml
name: 'Nightly Merge'

on:
  push:
    branches:
      - master

jobs:
  nightly-merge:

    runs-on: ubuntu-latest

    steps:
    - name: Checkout
      uses: actions/checkout@v1

    - name: Nightly Merge
      uses: ./.github/actions/merge-branches
      with:
        source_branch: 'master'
        target_branch: 'nightly'
        allow_ff: true
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

Even though this action was created to run when a new commit is checked in to master,
[`on`](https://help.github.com/en/articles/workflow-syntax-for-github-actions#on)
can be replaced by any other trigger.
For example, this will run the action at a recurring scheduled time:

```yml
on:
  schedule:
    - cron:  '0 0 * * *'
```

The `Checkout` step checks out your repository so that the workflow
can access its contents.

## Parameters

### `source_branch`

The name of the source branch (default `master`).

### `target_branch`

The name of the target branch (default `devel`).

### `allow_ff`

Allow fast forward merge (default `false`). If not enabled, merges will use
`--no-ff`.

### `ff_only`

Refuse to merge and exit unless the current HEAD is already up to date or the
merge can be resolved as a fast-forward (default `false`).
Requires `allow_ff=true`.

### `allow_forks`

Allow action to run on forks (default `false`).

### `user_name`

User name for git commits (default `GitHub Nightly Merge Action`).

### `user_email`

User email for git commits (default `actions@github.com`).

### `push_token`

Environment variable containing the token to use for push (default
`GITHUB_TOKEN`).
Useful for pushing on protected branches.
Using a secret to store this variable value is strongly recommended, since this
value will be printed in the logs.
The `GITHUB_TOKEN` is still used for API calls, therefore both token should be
available.

```yml
      with:
        push_token: 'FOO_TOKEN'
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        FOO_TOKEN: ${{ secrets.FOO_TOKEN }}
```
