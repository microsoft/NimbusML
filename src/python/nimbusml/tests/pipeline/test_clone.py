# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
import unittest

import numpy as np
import pandas as pd
from nimbusml import FileDataStream
from nimbusml import Pipeline, Role
from nimbusml.datasets import get_dataset
from nimbusml.ensemble import LightGbmRanker
from nimbusml.feature_extraction.categorical import OneHotVectorizer
from nimbusml.linear_model import FastLinearRegressor
from nimbusml.preprocessing import ToKey
from nimbusml.utils.exports import dot_export_pipeline

# set up dataframe
path = get_dataset("gen_tickettrain").as_filepath()
df = pd.read_csv(path)
df = df.rename(index=str, columns={'rank': 'label_1', 'group': 'group_2'})
features = ['price', 'Class', 'duration']
df['group_2'] = df['group_2'].astype(np.uint32)
X = df.drop(['label_1'], axis=1)
y = df['label_1']

# set up filedatastream
fds = FileDataStream.read_csv(path, names={0: 'label_1', 1: 'group_2'})


def clone_and_check(pipe):
    pipe_attrs = pipe.__dict__.copy()
    cloned_pipe = pipe.clone()
    cloned_attrs = cloned_pipe.__dict__.copy()
    assert pipe_attrs.__repr__() == cloned_attrs.__repr__()


def print_debug(title, pipe):
    print("--", title, "--")
    di = pipe if isinstance(pipe, dict) else pipe.steps[-1].__dict__
    for k, v in sorted(di.items()):
        if isinstance(v, dict):
            for kk, vv in sorted(v.items()):
                if isinstance(vv, list):
                    print("{2}.{0}={1}".format(
                        kk, vv[:5] + [".."] if len(vv) > 5 else vv, k))
                else:
                    print("{2}.{0}={1}".format(kk, vv, k))
        elif isinstance(v, list):
            print("{0}={1}".format(k, v[:5] + [".."] if len(v) > 5 else v))
        else:
            print("{0}={1}".format(k, v))


def print_debug_diff(title, pipe1, pipe2):
    print("--", title, "--")
    di1 = pipe1 if isinstance(pipe1, dict) else pipe1.steps[-1].__dict__
    di2 = pipe2 if isinstance(pipe2, dict) else pipe2.steps[-1].__dict__
    for k, v in sorted(di1.items()):
        if k not in di2:
            message = " -"
            v2 = ""
        else:
            message = "!=" if v != di2[k] else "=="
            v2 = di2[k] if message != "==" else ""
        if isinstance(v, dict):
            for kk, vv in sorted(v.items()):
                if isinstance(vv, list):
                    print("{3} {2}.{0}={1} -- {4}".format(kk,
                                                          vv[:5] + [
                                                              ".."] if len(
                                                              vv) > 5 else vv,
                                                          k,
                                                          message,
                                                          v2.get(kk,
                                                                 "?")))
                else:
                    print("{3} {2}.{0}={1} -- {4}".format(kk,
                                                          vv, k, message,
                                                          v2.get(kk, "?")))
        elif isinstance(v, list):
            print("{2} {0}={1} -- {3}".format(k,
                                              v[:5] + [".."] if len(
                                                  v) > 5 else v, message, v2))
        else:
            print("{2} {0}={1} -- {3}".format(k, v, message, v2))
    for k, v in sorted(di2.items()):
        if k not in di1:
            message = " +"
        else:
            continue
        if isinstance(v, dict):
            for kk, vv in sorted(v.items()):
                if isinstance(vv, list):
                    print("{3} {2}.{0}={1}".format(
                        kk, vv[:5] + [".."] if len(vv) > 5 else vv, k,
                        message))
                else:
                    print("{3} {2}.{0}={1}".format(kk, vv, k, message))
        elif isinstance(v, list):
            print("{2} {0}={1}".format(
                k, v[:5] + [".."] if len(v) > 5 else v, message))
        else:
            print("{2} {0}={1}".format(k, v, message))


def copy_shorten_dict(d):
    cpy = d.copy()
    for k in cpy:
        if isinstance(cpy[k], list) and len(cpy[k]) > 5:
            cpy[k] = cpy[k][:5] + ['...']
        elif isinstance(cpy[k], dict):
            cpy[k] = copy_shorten_dict(cpy[k])
    return cpy


def fit_test_clone_and_check(pipe, data, debug=False):
    if debug:
        import pprint
        print("------------ pipe.steps[-1]")
        pprint.pprint(copy_shorten_dict(pipe.steps[-1].__dict__))
    assert getattr(pipe.steps[-1], "input", None) is None
    assert getattr(pipe.steps[-1], "output", None) is None
    cloned_pipe = pipe.clone()
    if debug:
        import pprint
        print("------------ cloned_pipe.steps[-1] cloned")
        pprint.pprint(copy_shorten_dict(cloned_pipe.steps[-1].__dict__))
    # print_debug("not cloned", pipe)
    # print_debug("cloned", cloned_pipe)
    assert getattr(cloned_pipe.steps[-1], "input", None) is None
    assert getattr(cloned_pipe.steps[-1], "output", None) is None
    roles1 = pipe.steps[-1].get_roles_params()
    pipe_attrs = pipe.__dict__.copy()
    pipe.fit(data, verbose=0)
    if debug:
        import pprint
        print("------------ pipe.steps[-1] fitted")
        pprint.pprint(copy_shorten_dict(pipe.steps[-1].__dict__))
    roles2 = pipe.steps[-1].get_roles_params()
    metrics, _ = pipe.test(data)
    sum1 = metrics.sum().sum()
    cloned_pipe = pipe.clone()
    assert getattr(cloned_pipe.steps[-1], "input", None) is None
    assert getattr(cloned_pipe.steps[-1], "output", None) is None
    roles3 = cloned_pipe.steps[-1].get_roles_params()
    cloned_attrs = cloned_pipe.__dict__.copy()
    # print_debug("cloned", cloned_pipe)
    # print_debug_diff("diff", pipe_attrs["steps"][-1].__dict__, cloned_pipe)
    cloned_pipe.fit(data, verbose=0)
    roles4 = cloned_pipe.steps[-1].get_roles_params()
    metrics, _ = cloned_pipe.test(data)
    roles5 = cloned_pipe.steps[-1].get_roles_params()
    sum2 = metrics.sum().sum()
    assert sum1 > 1
    assert sum1 == sum2
    assert roles1 == roles3
    assert roles2 == roles4
    assert roles4 == roles5
    assert pipe_attrs.__repr__() == cloned_attrs.__repr__()


def fit_transform_clone_and_check(pipe, data):
    pipe_attrs = pipe.__dict__.copy()
    outdata1 = pipe.fit_transform(data)
    cloned_pipe = pipe.clone()
    cloned_attrs = cloned_pipe.__dict__.copy()
    outdata2 = cloned_pipe.fit_transform(data)
    assert str(outdata1) == str(outdata2)
    assert pipe_attrs.__repr__() == cloned_attrs.__repr__()


class TestPipelineClone(unittest.TestCase):

    def test_nofit_pipeline_clone(self):
        pipe = Pipeline([
            LightGbmRanker(feature=features,
                           label='label_1',
                           group_id='group_2',
                           number_of_iterations=1,
                           number_of_leaves=4)
        ])
        clone_and_check(pipe)

    def test_pipeline_clone_dataframe_roles_arguments(self):
        pipe = Pipeline([
            LightGbmRanker(feature=features,
                           label='label_1',
                           group_id='group_2',
                           number_of_iterations=1,
                           number_of_leaves=4)
        ])
        fit_test_clone_and_check(pipe, df)

    def test_pipeline_clone_dataframe_roles_shift_operator(self):
        pipe = Pipeline([
            LightGbmRanker(number_of_iterations=1, number_of_leaves=4) << {
                Role.Feature: features,
                Role.Label: 'label_1',
                Role.GroupId: 'group_2'}
        ])
        fit_test_clone_and_check(pipe, df, debug=False)

    def test_pipeline_clone_filedatastream_roles_arguments(self):
        pipe = Pipeline([
            ToKey() << {'group_2': 'group_2'},
            LightGbmRanker(feature=features,
                           label='label_1',
                           group_id='group_2',
                           number_of_iterations=1,
                           number_of_leaves=4)
        ])
        fit_test_clone_and_check(pipe, fds)

    def test_pipeline_clone_filedatastream_roles_shift_operator(self):
        pipe = Pipeline([
            ToKey() << {'group_2': 'group_2'},
            LightGbmRanker(number_of_iterations=1, number_of_leaves=4) << {
                Role.Feature: features,
                Role.Label: 'label_1',
                Role.GroupId: 'group_2'}
        ])
        fit_test_clone_and_check(pipe, fds)

    def test_pipeline_clone_dataframe_transforms(self):
        pipe = Pipeline([
            OneHotVectorizer(columns={'onehot': 'group_2'})
        ])
        fit_transform_clone_and_check(pipe, df)

    def test_pipeline_clone_dataframe_transforms_shift_operator(self):
        pipe = Pipeline([
            OneHotVectorizer() << {'onehot': 'group_2'}
        ])
        fit_transform_clone_and_check(pipe, df)

    def test_pipeline_clone_filedatastream_transforms(self):
        pipe = Pipeline([
            OneHotVectorizer(columns={'onehot': 'group_2'})
        ])
        fit_transform_clone_and_check(pipe, fds)

    def test_pipeline_clone_filedatastream_transforms_shift_operator(self):
        pipe = Pipeline([
            OneHotVectorizer() << {'onehot': 'group_2'}
        ])
        fit_transform_clone_and_check(pipe, fds)

    def test_good_state_after_raise(self):
        df = pd.DataFrame(dict(edu=[1.12, 1.2, 2.24, 4.4, 5.4],
                               wk=[1.1, 2.3, 1.56, 0.4, 2.4],
                               y=[1.1, 2.2, 1.24, 3.4, 3.4]))

        lr = FastLinearRegressor()
        pipe = Pipeline([lr])

        assert not pipe.nodes[0].has_defined_columns()
        assert not pipe._is_fitted
        with self.assertRaises(RuntimeError):
            pipe.fit(df, verbose=0)
        assert not pipe._is_fitted
        assert pipe._run_time_error is not None
        pipe.fit(df[['edu', 'wk']], df['y'], verbose=0)
        assert pipe._is_fitted

    def test_plot_fitted_cloned_pipeline(self):
        df = pd.DataFrame(dict(education=['A', 'B', 'A', 'B', 'A'],
                               workclass=['X', 'X', 'Y', 'Y', 'Y'],
                               y=[1.0, 3, 2, 3, 4]))
        exp = Pipeline([
            OneHotVectorizer() << ['workclass', 'education'],
            FastLinearRegressor(feature=['workclass', 'education'], label='y'),
        ])
        info1 = exp.get_fit_info(df)[0]
        res1 = dot_export_pipeline(exp, df)
        assert res1 is not None
        exp.fit(df)
        info2 = exp.get_fit_info(df)[0]
        assert len(info1) == len(info2)
        exp.fit(df)
        info3 = exp.get_fit_info(df)[0]
        assert len(info1) == len(info3)

        for i, (a, b, c) in enumerate(zip(info1, info2, info3)):
            assert list(sorted(a)) == list(sorted(b))
            assert list(sorted(a)) == list(sorted(c))
            for k in sorted(a):
                if not isinstance(a[k], (list, dict, str, int, float, tuple)):
                    continue
                if b[k] != c[k]:
                    import pprint
                    pprint.pprint(b)
                    pprint.pprint(c)
                    raise Exception(
                        "Issue with "
                        "op={0}\nk='{1}'\n---\n{2}\n---\n{3}".format(
                            i, k, b[k], c[k]))
                if a[k] != b[k]:
                    import pprint
                    pprint.pprint(a)
                    pprint.pprint(b)
                    raise Exception(
                        "Issue with "
                        "op={0}\nk='{1}'\n---\n{2}\n---\n{3}".format(
                            i, k, a[k], b[k]))
        res2 = dot_export_pipeline(exp, df)
        assert res2 is not None
        assert res1 == res2


if __name__ == '__main__':
    unittest.main()
