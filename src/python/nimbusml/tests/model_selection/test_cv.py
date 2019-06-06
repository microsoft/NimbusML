# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import os
import unittest

import numpy as np
import pandas as pd
from nimbusml import Pipeline, FileDataStream, Role, DataSchema
from nimbusml.cluster import KMeansPlusPlus
from nimbusml.datasets import get_dataset
from nimbusml.ensemble import FastForestRegressor, LightGbmRanker
from nimbusml.feature_extraction.categorical import OneHotVectorizer, \
    OneHotHashVectorizer
from nimbusml.linear_model import FastLinearClassifier, \
    LogisticRegressionBinaryClassifier, LogisticRegressionClassifier
from nimbusml.model_selection import CV
from nimbusml.preprocessing import ToKey
from nimbusml.preprocessing.schema import ColumnConcatenator, ColumnDropper
from nimbusml.tests.test_utils import split_features_and_label
from sklearn.utils.testing import assert_equal, assert_true, \
    assert_greater_equal

infert_file = get_dataset('infert').as_filepath()


def default_pipeline(
        learner=FastForestRegressor,
        transforms=[],
        learner_arguments={},
        init_pipeline=True):
    pipeline = transforms + [learner(**learner_arguments)]
    if init_pipeline:
        pipeline = Pipeline(pipeline)
    return pipeline


def infert_ds(label_index, label_name='Label'):
    file_schema = 'sep=, col=id:TX:0 col=education:TX:1 col={}:R4:{} ' \
                  'col=Features:R4:{}-8 header=+'.format(
                      label_name, label_index, label_index + 1)
    data = FileDataStream(infert_file, schema=file_schema)
    if label_name != 'Label':
        data._set_role(Role.Label, label_name)
    return data


def infert_df(label_name):
    df = get_dataset('infert').as_df()
    df = (OneHotVectorizer() << 'education_str').fit_transform(df)
    X, y = split_features_and_label(df, label_name)
    return X, y


def random_df(shape=(100, 3)):
    np.random.seed(0)
    df = pd.DataFrame(np.random.rand(*shape))
    df.columns = df.columns.astype('str')
    return df


def random_series(values=[0, 1], length=100, type='int', name=None):
    np.random.seed(0)
    return pd.Series(np.random.choice(values, length).astype(type), name=name)


def default_infert_transforms():
    return [OneHotVectorizer(columns={'edu': 'education'})]


def default_infert_learner_arguments():
    return {'feature': ['Features', 'edu']}


def check_cv_results(learner_type, results, n_folds, expected_metrics):
    assert_true(isinstance(results, dict))
    result_names = set(results.keys())
    common_outputs = {'predictions', 'models', 'metrics', 'metrics_summary'}
    classification_outputs = common_outputs.union({'confusion_matrix'})
    if learner_type in ['regressor', 'ranker', 'clusterer']:
        assert_equal(result_names, common_outputs)
    elif learner_type in ['binary', 'multiclass']:
        assert_equal(result_names, classification_outputs)
    else:
        assert_true(False, 'Invalid learner type ' + learner_type)

    for name, df in results.items():
        if name == 'metrics_summary':
            # metrics_summary has no fold column
            # check for metrics accuracy
            for m_name, m_expected_value in expected_metrics.items():
                m_value = df.loc['Average', m_name]
                assert_greater_equal(
                    m_value,
                    m_expected_value,
                    msg='Metric {} is lower than expected'.format(m_name))

            # no more checks for metrics_summary
            continue

        assert_true(CV.fold_column_name in df.columns)
        folds_series = df[CV.fold_column_name]

        if name in ['models', 'metrics']:
            folds_count = len(folds_series)
        else:
            folds_count = pd.Series.nunique(folds_series)

        assert_equal(folds_count, n_folds)


def check_cv(
        pipeline,
        X,
        y=None,
        n_folds=2,
        groups=None,
        split_start='before_transforms',
        expected_metrics={},
        **params):
    cv = CV(pipeline)
    if split_start == 'try_all':
        len_pipeline = len(pipeline.nodes)
        values_to_test = ['after_transforms', 'before_transforms']
        values_to_test.extend(list(range(len_pipeline)))
        values_to_test.extend(list(range(-len_pipeline, 0)))
        for s in values_to_test:
            graph_id = '_split_start={}'.format(str(s))
            results = cv.fit(
                X,
                y,
                cv=n_folds,
                groups=groups,
                split_start=s,
                graph_id=graph_id)
            check_cv_results(
                cv._learner_type,
                results,
                n_folds,
                expected_metrics)
    else:
        results = cv.fit(
            X,
            y,
            cv=n_folds,
            groups=groups,
            split_start=split_start,
            **params)
        check_cv_results(cv._learner_type, results, n_folds, expected_metrics)

    return results


class TestCvRegressor(unittest.TestCase):
    infert_age_index = 2

    def pipeline(self, **params):
        return default_pipeline(learner=FastForestRegressor, **params)

    def default_pipeline(self, init_pipeline=True):
        return self.pipeline(
            transforms=default_infert_transforms(),
            learner_arguments=default_infert_learner_arguments(),
            init_pipeline=init_pipeline)

    def data(self, label_name):
        return infert_ds(self.infert_age_index, label_name)

    def check_cv_with_defaults(
            self,
            label_name='age',
            init_pipeline=True,
            **params):
        check_cv(
            self.default_pipeline(init_pipeline),
            self.data(label_name),
            **params)

    def test_default_label(self):
        expected_metrics = {'L1(avg)': 3.670149, 'L2(avg)': 20.576492}
        self.check_cv_with_defaults(
            'Label',
            split_start='try_all',
            expected_metrics=expected_metrics)

    def test_non_default_label(self):
        self.check_cv_with_defaults()

    def test_num_folds(self):
        self.check_cv_with_defaults(n_folds=5)
        self.check_cv_with_defaults(n_folds=10)

    def test_no_transform(self):
        check_cv(
            self.pipeline(
                learner_arguments={
                    'feature': 'Features'}),
            self.data('age'),
            split_start='try_all')

    def test_df(self):
        X, y = infert_df('age')
        check_cv(self.pipeline(), X, y)

    # The following tests are only needed once. We bundle them with regressor
    # tests.
    def test_pipeline_steps_syntax(self):
        self.check_cv_with_defaults(init_pipeline=False)

    def test_unsupported_pipelines(self):
        unsupported_pipelines = [OneHotVectorizer()]
        for p in unsupported_pipelines:
            pipeline = Pipeline([p])
            msg = 'CV doesn\'t support pipeline ending with {}, but ' \
                  'didn\'t raise exception'.format(
                      pipeline._last_node_type())
            with self.assertRaises(ValueError, msg=msg):
                CV(pipeline)

    def test_split_start(self):
        long_transforms = [
            OneHotVectorizer(
                columns={
                    'edu': 'education'}), OneHotHashVectorizer(
                columns={
                    'edu_hash': 'education'}), ColumnDropper(
                columns='education')]
        pipeline = self.pipeline(
            transforms=long_transforms,
            learner_arguments={
                'feature': [
                    'Features',
                    'edu',
                    'edu_hash']})
        check_cv(pipeline, self.data('Label'), split_start='try_all')

    def test_unsupported_split_start(self):
        pipeline = self.default_pipeline()
        pip_len = len(pipeline.nodes)
        unsupported_values = ['random_string', pip_len,
                              pip_len + 1, -pip_len - 1, -pip_len - 2]
        for split_start in unsupported_values:
            msg = 'Split_start={} is invalid, but CV didn\'t raise ' \
                  'exception'.format(
                      split_start)
            with self.assertRaises(ValueError, msg=msg):
                self.check_cv_with_defaults(
                    split_start=split_start, graph_id=str(split_start))


class TestCvBinary(unittest.TestCase):
    infert_case_index = 5

    def pipeline(self, **params):
        return default_pipeline(
            learner=LogisticRegressionBinaryClassifier, **params)

    def data(self, label_name):
        return infert_ds(self.infert_case_index, label_name)

    def check_cv_with_defaults(self, label_name='case', **params):
        pipeline = self.pipeline(
            transforms=default_infert_transforms(),
            learner_arguments=default_infert_learner_arguments())
        check_cv(pipeline, self.data(label_name), **params)

    def test_default_label(self):
        expected_metrics = {'AUC': 0.662212, 'Accuracy': 0.703976}
        self.check_cv_with_defaults(
            'Label',
            split_start='try_all',
            expected_metrics=expected_metrics)

    def test_non_default_label(self):
        self.check_cv_with_defaults()

    def test_num_folds(self):
        self.check_cv_with_defaults(n_folds=3)
        self.check_cv_with_defaults(n_folds=6)

    def test_no_transform(self):
        check_cv(
            self.pipeline(
                learner_arguments={
                    'feature': 'Features'}),
            self.data('case'),
            split_start='try_all')

    def test_df(self):
        X, y = infert_df('case')
        check_cv(self.pipeline(), X, y)

    def test_groups(self):
        # one learner type is enough for testing sanity of groups argument
        file_schema = 'sep=, col=age:TX:2 col=Label:R4:5 ' \
                      'col=Features:R4:6-8 header=+'
        data = FileDataStream(infert_file, schema=file_schema)
        expected_metrics = {'AUC': 0.704883, 'Accuracy': 0.717414}
        pipeline = self.pipeline(
            learner_arguments={
                'feature': 'Features'},
            transforms=[])
        check_cv(
            pipeline,
            data,
            groups='age',
            expected_metrics=expected_metrics)


class TestCvMulticlass(unittest.TestCase):
    infert_induced_index = 4

    def pipeline(self, **params):
        return default_pipeline(learner=LogisticRegressionClassifier, **params)

    def data(self, label_name):
        return infert_ds(self.infert_induced_index, label_name)

    def check_cv_with_defaults(self, label_name='induced', **params):
        pipeline = self.pipeline(
            transforms=default_infert_transforms(),
            learner_arguments=default_infert_learner_arguments())
        check_cv(pipeline, self.data(label_name), **params)

    def test_default_label(self):
        expected_metrics = {
            'Accuracy(micro-avg)': 0.580443,
            'Accuracy(macro-avg)': 0.339080}
        self.check_cv_with_defaults(
            'Label',
            split_start='try_all',
            expected_metrics=expected_metrics)

    def test_non_default_label(self):
        self.check_cv_with_defaults()

    def test_num_folds(self):
        self.check_cv_with_defaults(n_folds=5)
        self.check_cv_with_defaults(n_folds=7)

    def test_no_transform(self):
        check_cv(
            self.pipeline(
                learner_arguments={
                    'feature': 'Features'}),
            self.data('induced'),
            split_start='try_all')

    def test_df(self):
        X, y = infert_df('induced')
        check_cv(self.pipeline(), X, y)

    def test_unseen_classes(self):
        # Create a dataset such that cv splits miss some of the classes
        X = random_df()
        y = random_series()
        y[95:] = range(5)

        msg = 'CV didn\'t raise Warning exception b/c of minority class issue'
        with self.assertRaises(Warning, msg=msg):
            cv = CV([FastLinearClassifier()])
            cv.fit(X, y, cv=3)


class TestCvRanker(unittest.TestCase):
    def data(self, label_name, group_id, features):
        simpleinput_file = get_dataset("gen_tickettrain").as_filepath()
        file_schema = 'sep=, col={label}:R4:0 col={group_id}:TX:1 ' \
                      'col={features}:R4:3-5'.format(
                        label=label_name, group_id=group_id, features=features)
        data = FileDataStream(simpleinput_file, schema=file_schema)
        if label_name != 'Label':
            data._set_role(Role.Label, label_name)
        return data

    def data_pandas(self):
        simpleinput_file = get_dataset("gen_tickettrain").as_filepath()
        data = pd.read_csv(simpleinput_file)
        data['group'] = data['group'].astype(str)
        return data

    def data_wt_rename(self, label_name, group_id, features):
        simpleinput_file = get_dataset("gen_tickettrain").as_filepath()
        file_schema = 'sep=, col={label}:R4:0 col={group_id}:TX:1 ' \
                      'col={features}:R4:3-5'.format(
                        label=label_name, group_id=group_id, features=features)
        data = FileDataStream(simpleinput_file, schema=file_schema)
        if label_name != 'Label':
            data._set_role(Role.Label, label_name)
        return data

    @unittest.skipIf(os.name != "nt", "random crashes on linux")
    def check_cv_with_defaults2(
            self,
            label_name='Label',
            group_id='GroupId',
            features='Features_1',
            **params):
        # REVIEW: Replace back ToKey() with OneHotHashVectorizer()  and reinstate metrics checks
        # once issue https://github.com/dotnet/machinelearning/issues/1939 is resolved. 
        params.pop('expected_metrics', None)
        steps = [ToKey() << {
                group_id: group_id}, ColumnConcatenator() << {
                'Features': [features]}, LightGbmRanker(
                minimum_example_count_per_leaf=1) << {
                Role.GroupId: group_id}]
        data = self.data_wt_rename(label_name, group_id, features)
        check_cv(pipeline=Pipeline(steps), X=data, **params)

    @unittest.skipIf(os.name != "nt", "random crashes on linux")
    def check_cv_with_defaults_df(
            self,
            label_name='rank',
            group_id='group',
            features=['price','Class','dep_day','nbr_stops','duration'],
            **params):
        steps = [
            ToKey() << {
                group_id: group_id},
            LightGbmRanker(
                minimum_example_count_per_leaf=1,
                feature=features,
                label='rank', group_id='group'
            )]
        data = self.data_pandas()
        check_cv(pipeline=Pipeline(steps), X=data, **params)

    @unittest.skipIf(os.name != "nt", "random crashes on linux")
    def test_default_df(self):
        self.check_cv_with_defaults_df()

    @unittest.skipIf(os.name != "nt", "random crashes on linux")
    def test_default_label2(self):
        self.check_cv_with_defaults2(split_start='try_all')

    @unittest.skipIf(os.name != "nt", "random crashes on linux")
    def test_metrics2(self):
        expected_metrics = {
            'NDCG@1': 64.746031,
            'NDCG@2': 68.117459,
            'NDCG@3': 72.163466,
            'DCG@1': 5.083095,
            'DCG@2': 7.595355,
            'DCG@3': 8.905803}
        self.check_cv_with_defaults2(
            n_folds=5, expected_metrics=expected_metrics)

    @unittest.skipIf(os.name != "nt", "random crashes on linux")
    def test_non_default_label2(self):
        self.check_cv_with_defaults2(label_name='Label_1')

    @unittest.skipIf(os.name != "nt", "random crashes on linux")
    def test_non_default_group_id2(self):
        self.check_cv_with_defaults2(
            label_name='Label_1', group_id='GroupId_1')

    @unittest.skipIf(os.name != "nt", "random crashes on linux")
    def test_num_folds2(self):
        self.check_cv_with_defaults2(n_folds=3)

    @unittest.skipIf(os.name != "nt", "random crashes on linux")
    def check_cv_with_defaults(
            self,
            label_name='Label',
            group_id='GroupId',
            features='Features_1',
            **params):
        # REVIEW: Replace back ToKey() with OneHotHashVectorizer()  and reinstate metrics checks
        # once issue https://github.com/dotnet/machinelearning/issues/1939 is resolved. 
        params.pop('expected_metrics', None)
        steps = [ToKey() << {
                     group_id: group_id},
                 # even specify all the roles needed in the following line, the
                 # roles are still not passed correctly
                 LightGbmRanker(minimum_example_count_per_leaf=1) << {
                     Role.GroupId: group_id, Role.Feature: features,
                     Role.Label: label_name}]
        data = self.data(label_name, group_id, features)
        check_cv(pipeline=Pipeline(steps), X=data, **params)

    @unittest.skipIf(os.name != "nt", "random crashes on linux")
    def test_default_label(self):
        self.check_cv_with_defaults(split_start='try_all')

    @unittest.skipIf(os.name != "nt", "random crashes on linux")
    def test_metrics(self):
        expected_metrics = {
            'NDCG@1': 64.746031,
            'NDCG@2': 68.117459,
            'NDCG@3': 72.163466,
            'DCG@1': 5.083095,
            'DCG@2': 7.595355,
            'DCG@3': 8.905803}
        self.check_cv_with_defaults(
            n_folds=5, expected_metrics=expected_metrics)

    @unittest.skipIf(os.name != "nt", "random crashes on linux")
    def test_non_default_label(self):
        self.check_cv_with_defaults(label_name='Label_1')

    @unittest.skipIf(os.name != "nt", "random crashes on linux")
    def test_non_default_group_id(self):
        self.check_cv_with_defaults(label_name='Label_1', group_id='GroupId_1')

    @unittest.skipIf(os.name != "nt", "random crashes on linux")
    def test_num_folds(self):
        self.check_cv_with_defaults(n_folds=3)

    def check_cv_with_non_defaults(
            self,
            label_name='label',
            group_id='groupid',
            features='Features_1',
            **params):
        steps = [
            ToKey(
                columns={
                    'groupid2': group_id,
                    'label2': label_name}),
            LightGbmRanker() << {
                Role.GroupId: 'groupid2',
                Role.Label: 'label2',
                Role.Feature: [features]}]
        data = self.data(label_name, group_id, features)
        cv = CV(steps)
        results = cv.fit(data, groups='groupid', cv=4)
        check_cv_results(
            cv._learner_type,
            results,
            n_folds=4,
            expected_metrics={})

    def test_non_default_label3(self):
        self.check_cv_with_non_defaults(split_start='try_all')


class TestCvClusterer(unittest.TestCase):
    def test_defaults(self):
        schema = DataSchema.read_schema(infert_file, numeric_dtype=np.float32)
        data = FileDataStream.read_csv(infert_file, schema=schema)
        pipeline_steps = [
            OneHotVectorizer(
                columns={
                    'edu': 'education'}),
            KMeansPlusPlus(
                n_clusters=5,
                feature=[
                    'edu',
                    'age',
                    'parity',
                    'spontaneous',
                    'stratum'])]
        check_cv(pipeline_steps, data)

    def test_df(self):
        # Define 3 clusters with centroids (1,1,1), (11,11,11) and
        # (-11,-11,-11)
        X = pd.DataFrame(data=dict(
            x=[0, 1, 2, 10, 11, 12, -10, -11, -12],
            y=[0, 1, 2, 10, 11, 12, -10, -11, -12],
            z=[0, 1, 2, 10, 11, 12, -10, -11, -12]))
        check_cv([KMeansPlusPlus(n_clusters=3)], X)
