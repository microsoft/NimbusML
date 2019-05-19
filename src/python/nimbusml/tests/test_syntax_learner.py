# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
"""
test low-level entrypoints
"""
import unittest

import pandas
from nimbusml import Pipeline
from nimbusml.datasets import get_dataset
from nimbusml.ensemble import FastTreesRegressor
from nimbusml.feature_extraction.categorical import OneHotVectorizer
from nimbusml.internal.utils.data_roles import Role
from nimbusml.linear_model import AveragedPerceptronBinaryClassifier
from nimbusml.linear_model import FastLinearBinaryClassifier, \
    FastLinearRegressor, OnlineGradientDescentRegressor
from nimbusml.preprocessing import ToKey
from nimbusml.preprocessing.normalization import MeanVarianceScaler
from nimbusml.preprocessing.schema import ColumnConcatenator as Concat, \
    ColumnDropper as Drop
from nimbusml.preprocessing.schema import TypeConverter


class TestSyntaxLearner(unittest.TestCase):
    @unittest.skip("TypeConverter does not seem to work here.")
    def test_syntax7(self):
        # Error message are usually not informative enough.
        # Missing column --> no indication of other columns.
        # Error is (one transform should handle it)
        # 'The label column 'y' of the training data has a data type
        # not suitable for binary classification: Vec<Key<U4, 0-1>, 2>.
        # Type must be Bool, R4, R8 or Key with two classes.

        df = pandas.DataFrame(
            dict(
                education=[
                    'A', 'B', 'A', 'B', 'A'], workclass=[
                    'X', 'X', 'Y', 'Y', 'Y'], y=[
                    'red', 'white', 'red', 'white', 'white']))
        X = df.drop('y', axis=1)
        y = df['y']

        exp = Pipeline([
            OneHotVectorizer() << 'y',
            OneHotVectorizer() << ['workclass', 'education'],
            TypeConverter(result_type='R4') << 'y',
            FastLinearBinaryClassifier(maximum_number_of_iterations=1)
        ])
        exp.fit(X, y, verbose=0)
        prediction = exp.predict(X)
        assert isinstance(prediction, pandas.DataFrame)
        assert list(
            prediction.columns) == [
            'PredictedLabel',
            'Probability',
            'Score']
        assert prediction.shape == (5, 3)
        assert prediction.min() > 0.01
        assert prediction.max() < 0.05

    @unittest.skip("type conversion does not work")
    def test_syntax7_rename(self):
        # Error message are usually not informative enough.
        # Missing column --> no indication of other columns.
        # Error is (one transform should handle it)
        # 'The label column 'y' of the training data has a data type
        # not suitable for binary classification: Vec<Key<U4, 0-1>, 2>.
        # Type must be Bool, R4, R8 or Key with two classes.

        df = pandas.DataFrame(
            dict(
                education=[
                    'A', 'B', 'A', 'B', 'A'], workclass=[
                    'X', 'X', 'Y', 'Y', 'Y'], y=[
                    'red', 'white', 'red', 'white', 'white']))
        X = df.drop('y', axis=1)
        y = df['y']

        exp = Pipeline([
            OneHotVectorizer() << 'y',
            OneHotVectorizer() << ['workclass', 'education'],
            TypeConverter(result_type='R4') << {'yi': 'y'},
            Drop() << 'y',
            FastLinearBinaryClassifier(maximum_number_of_iterations=1) << 'yi'
        ])
        exp.fit(X, y, verbose=0)
        prediction = exp.predict(X)
        assert isinstance(prediction, pandas.DataFrame)
        assert list(prediction.columns) == ['Score']
        assert prediction.shape == (5, 1)
        assert prediction.min() > 0.01
        assert prediction.max() < 0.05

    def test_syntax8_label(self):
        df = pandas.DataFrame(dict(education=['A', 'B', 'A', 'B', 'A'],
                                   workclass=['X', 'X', 'Y', 'Y', 'Y'],
                                   yy=[1.1, 2.2, 1.24, 3.4, 3.4]))
        X = df.drop('yy', axis=1)

        exp = Pipeline([
            MeanVarianceScaler() << {'new_y': 'yy'},
            OneHotVectorizer() << ['workclass', 'education'],
            Drop() << 'yy',
            FastLinearRegressor() << {'Feature': ['workclass', 'education'],
                                      Role.Label: 'new_y'}
        ])
        exp.fit(df, verbose=0)
        assert exp.nodes[-1].feature_column_name_ == 'Features'
        assert exp.nodes[-1].label_column_name_ == 'new_y'
        # The pipeline requires it now as it is transformed all along.
        X['yy'] = 0.0
        prediction = exp.predict(X, verbose=0)
        assert isinstance(prediction, pandas.DataFrame)
        assert list(prediction.columns) == ['Score']
        assert prediction.shape == (5, 1)
        if prediction['Score'].min() < 0.4:
            raise Exception(prediction)
        if prediction['Score'].max() > 2.00:
            raise Exception(prediction)

    def test_syntax9_label_name(self):
        X = pandas.DataFrame(dict(education=['A', 'B', 'A', 'B', 'A'],
                                  workclass=['X', 'X', 'Y', 'Y', 'Y'],
                                  yy=[1.1, 2.2, 1.24, 3.4, 3.4]))

        exp = Pipeline([
            MeanVarianceScaler() << {'new_y': 'yy'},
            OneHotVectorizer() << ['workclass', 'education'],
            Drop() << 'yy',
            FastLinearRegressor() << {'Feature': ['workclass', 'education'],
                                      Role.Label: 'new_y'}
        ])
        exp.fit(X, verbose=0)
        assert exp.nodes[-1].feature_column_name_ == 'Features'
        assert exp.nodes[-1].label_column_name_ == 'new_y'
        # The pipeline requires it now as it is transformed all along.
        X['yy'] = 0.0
        prediction = exp.predict(X)
        assert isinstance(prediction, pandas.DataFrame)
        assert list(prediction.columns) == ['Score']
        assert prediction.shape == (5, 1)
        if prediction['Score'].min() < 0.1:
            raise Exception(prediction)
        if prediction['Score'].max() > 2.0:
            raise Exception(prediction)

    def test_syntax10_weights_fail(self):
        df = pandas.DataFrame(dict(education=['A', 'B', 'A', 'B', 'A'],
                                   workclass=['X', 'X', 'Y', 'Y', 'Y'],
                                   weights=[1., 1., 1., 2., 1.],
                                   y=[1.1, 2.2, 1.24, 3.4, 3.4]))
        X = df.drop(['y', 'weights'], axis=1)
        y = df['y']
        weights = df['weights']

        exp = Pipeline([
            OneHotVectorizer() << ['workclass', 'education'],
            OnlineGradientDescentRegressor()
        ])
        try:
            exp.fit(X, y, weight=weights, verbose=0)
            assert False
        except RuntimeError as e:
            assert "does not support role 'Weight'" in str(e)

    @unittest.skip("Column 'weight' not found' --> weights not supported")
    def test_syntax10_weights(self):
        df = pandas.DataFrame(dict(education=['A', 'B', 'A', 'B', 'A'],
                                   workclass=['X', 'X', 'Y', 'Y', 'Y'],
                                   weight=[1., 1., 1., 2., 1.],
                                   y=[1.1, 2.2, 1.24, 3.4, 3.4]))
        X = df.drop(['y', 'weight'], axis=1)
        y = df['y']
        w = df['weight']

        exp = Pipeline([
            OneHotVectorizer() << ['workclass', 'education'],
            FastLinearRegressor()
        ])
        exp.fit(X, y, weight=w, verbose=0)
        assert exp.nodes[-1].feature_column_name == 'Features'
        assert exp.nodes[-1].label_column_name == 'y'
        assert exp.nodes[-1].example_weight_column_name == 'weight'
        X['weight'] = -5
        prediction = exp.predict(X)
        assert isinstance(prediction, pandas.DataFrame)
        assert list(prediction.columns) == ['Score']
        assert prediction.shape == (5, 1)
        if prediction['Score'].min() < 1.:
            raise Exception(prediction)
        if prediction['Score'].max() > 3.6:
            raise Exception(prediction)
        if len(set(prediction['Score'])) < 4:
            raise Exception(prediction)

    def test_syntax10_weights_operator(self):
        X = pandas.DataFrame(dict(education=['A', 'B', 'A', 'B', 'A'],
                                  workclass=['X', 'X', 'Y', 'Y', 'Y'],
                                  weight=[10., 1., 1., 1., 1.],
                                  y=[1.1, 2.2, 1.24, 3.4, 3.4]))

        exp = Pipeline(
            [
                OneHotVectorizer() << [
                    'workclass',
                    'education'],
                Concat() << {
                    'Feature': [
                        'workclass',
                        'education']},
                FastTreesRegressor(
                    number_of_trees=5) << {
                    'Feature': 'Feature',
                    Role.Label: 'y',
                    Role.Weight: 'weight'}])
        exp.fit(X, verbose=0)
        assert exp.nodes[-1].feature_column_name_ == 'Feature'
        assert exp.nodes[-1].label_column_name_ == 'y'
        assert exp.nodes[-1].example_weight_column_name_ == 'weight'
        # y is required here as well as weight.
        # It is replaced by fakes values.
        # The test does not fail but the weight is not taken into account.
        X['y'] = -5
        X['weight'] = -5
        prediction = exp.predict(X)
        assert isinstance(prediction, pandas.DataFrame)
        assert list(prediction.columns) == ['Score']
        assert prediction.shape == (5, 1)

    def test_syntax11_constructor(self):
        X = pandas.DataFrame(dict(education=['A', 'B', 'A', 'B', 'A'],
                                  workclass=['X', 'X', 'Y', 'Y', 'Y'],
                                  weight=[10., 1., 1., 1., 1.],
                                  y=[1.1, 2.2, 1.24, 3.4, 3.4]))

        exp = Pipeline([
            OneHotVectorizer(columns=['workclass', 'education']),
            Concat(columns={'Feature': ['workclass', 'education']}),
            FastTreesRegressor(number_of_trees=5, feature='Feature', label='y',
                               weight='weight')
        ])
        exp.fit(X, verbose=0)
        assert exp.nodes[-1].feature_column_name_ == 'Feature'
        assert exp.nodes[-1].label_column_name_ == 'y'
        assert exp.nodes[-1].example_weight_column_name_ == 'weight'
        # y is required here as well as weight.
        # It is replaced by fakes values.
        # The test does not fail but the weight is not taken into account.
        X['y'] = -5
        X['weight'] = -5
        prediction = exp.predict(X)
        assert isinstance(prediction, pandas.DataFrame)
        assert list(prediction.columns) == ['Score']
        assert prediction.shape == (5, 1)

    def test_syntax12_mixed1(self):
        X = pandas.DataFrame(dict(education=['A', 'B', 'A', 'B', 'A'],
                                  workclass=['X', 'X', 'Y', 'Y', 'Y'],
                                  weight=[10., 1., 1., 1., 1.],
                                  y=[1.1, 2.2, 1.24, 3.4, 3.4]))

        exp = Pipeline([
            OneHotVectorizer(columns=['workclass', 'education']),
            Concat(columns={'Feature': ['workclass', 'education']}),
            FastTreesRegressor(number_of_trees=5, label='y',
                               weight='weight') << 'Feature'
        ])
        exp.fit(X, verbose=0)
        assert exp.nodes[-1].feature_column_name_ == 'Feature'
        assert exp.nodes[-1].label_column_name_ == 'y'
        assert exp.nodes[-1].example_weight_column_name_ == 'weight'
        # y is required here as well as weight.
        # It is replaced by fakes values.
        # The test does not fail but the weight is not taken into account.
        X['y'] = -5
        X['weight'] = -5
        prediction = exp.predict(X)
        assert isinstance(prediction, pandas.DataFrame)
        assert list(prediction.columns) == ['Score']
        assert prediction.shape == (5, 1)

    def test_syntax12_mixed2(self):
        X = pandas.DataFrame(dict(education=['A', 'B', 'A', 'B', 'A'],
                                  workclass=['X', 'X', 'Y', 'Y', 'Y'],
                                  weight=[10., 1., 1., 1., 1.],
                                  y=[1.1, 2.2, 1.24, 3.4, 3.4]))

        exp = Pipeline(
            [
                OneHotVectorizer(
                    columns=[
                        'workclass', 'education']),
                Concat(
                    columns={
                        'Feature': ['workclass', 'education']}),
                FastTreesRegressor(
                    number_of_trees=5, feature='Feature', weight='weight') << {
                    Role.Label: 'y'}])
        exp.fit(X, verbose=0)
        assert exp.nodes[-1].feature_column_name_ == 'Feature'
        assert exp.nodes[-1].label_column_name_ == 'y'
        assert exp.nodes[-1].example_weight_column_name_ == 'weight'
        # y is required here as well as weight.
        # It is replaced by fakes values.
        # The test does not fail but the weight is not taken into account.
        X['y'] = -5
        X['weight'] = -5
        prediction = exp.predict(X)
        assert isinstance(prediction, pandas.DataFrame)
        assert list(prediction.columns) == ['Score']
        assert prediction.shape == (5, 1)

    def test_syntax12_group(self):
        # This tests check that a learner raises an exception
        # if a role is not allowed by the entrypoint.
        X = pandas.DataFrame(dict(education=['A', 'B', 'A', 'B', 'A'],
                                  workclass=['X', 'X', 'Y', 'Y', 'Y'],
                                  gr=[0, 0, 1, 1, 1],
                                  y=[1.1, 2.2, 1.24, 3.4, 3.4]))
        exp = Pipeline([
            OneHotVectorizer(columns=['workclass', 'education']),
            Concat(columns={'Feature': ['workclass', 'education']}),
            ToKey() << 'gr',
            FastTreesRegressor(number_of_trees=5, feature='Feature',
                               group_id='gr') << {Role.Label: 'y'}
        ])
        exp.fit(X, verbose=0)
        assert not hasattr(exp.nodes[-1], 'feature_')
        assert not hasattr(exp.nodes[-1], 'group_id_')
        assert exp.nodes[-1].feature_column_name_ == 'Feature'
        assert exp.nodes[-1].label_column_name_ == 'y'
        # assert not hasattr(exp.nodes[-1], 'row_group_column_name_')
        assert not hasattr(exp.nodes[-1], 'group_id_column')
        assert not hasattr(exp.nodes[-1], 'groupid_column_')
        assert not hasattr(exp.nodes[-1], 'groupid_column')
        if not hasattr(exp.nodes[-1], 'row_group_column_name_'):
            raise AssertionError("Attribute not found: {0}".format(
                ", ".join(sorted(dir(exp.nodes[-1])))))
        assert exp.nodes[-1].row_group_column_name_ == 'gr'
        # y is required here as well as weight.
        # It is replaced by fakes values.
        # The test does not fail but the weight is not taken into account.
        X['y'] = -5
        X['weight'] = -5
        prediction = exp.predict(X)
        assert isinstance(prediction, pandas.DataFrame)
        assert list(prediction.columns) == ['Score']
        assert prediction.shape == (5, 1)

    def test_syntax12_fail(self):
        # This tests check that a learner raises an exception
        # if a role is not allowed by the entrypoint.
        X = pandas.DataFrame(dict(education=['A', 'B', 'A', 'B', 'A'],
                                  workclass=['X', 'X', 'Y', 'Y', 'Y'],
                                  weight=[10., 1., 1., 1., 1.],
                                  y=[1.1, 2.2, 1.24, 3.4, 3.4]))
        try:
            Pipeline([
                OneHotVectorizer(columns=['workclass', 'education']),
                Concat(columns={'Feature': ['workclass', 'education']}),
                FastLinearBinaryClassifier(feature='Feature',
                                           group_id='weight') << {
                    Role.Label: 'y'}
            ])
            Pipeline.fit(X)
            assert False
        except (RuntimeError, NameError) as e:
            exp = "Parameter 'group_id' is not allowed " \
                "for class 'FastLinearBinaryClassifier'"
            if exp not in str(e):
                raise e

    # Test fails but really should not.
    # The error message mentions a predictor but there is none.
    @unittest.skip(
        'This test should work but does not due to: System.FormatException: " \
        Predictor model must contain role mappings')
    def test_syntax_concat_slots(self):
        X = pandas.DataFrame(dict(education=['A', 'B', 'A', 'B', 'A'],
                                  workclass=['X', 'X', 'Y', 'Y', 'Y'],
                                  weight=[10., 1., 1., 1., 1.],
                                  y=[1.1, 2.2, 1.24, 3.4, 3.4]))

        exp = Pipeline([
            OneHotVectorizer() << ['workclass', 'education'],
            Concat() << {'newcol': ['workclass', 'education']},
        ])
        exp.fit(X, verbose=0)
        exp.predict(X)
        # here, continue with checking what happens with multi-level index

    def test_syntax_slots_wo_pipeline(self):
        # data
        df = get_dataset("infert").as_df()
        df = df.drop(['row_num', ], axis=1)
        X = df.drop('case', axis=1)
        y = df['case']

        # transform
        xf1 = OneHotVectorizer(columns=['age', 'parity', 'education_str'])
        X_xf1 = xf1.fit_transform(X, verbose=0)
        assert "age.21" in list(X_xf1.columns)

        # learner
        # (1 .a.)
        model = AveragedPerceptronBinaryClassifier()

        # (1. b)
        try:
            model = AveragedPerceptronBinaryClassifier(feature=['age'])
            model.fit(X_xf1, y, verbose=0)
            cont = True
            assert False
        except Exception as e:
            # does not work
            cont = False
            print(e)

        if cont:
            y_pred = model.predict(X_xf1)
            assert y_pred.shape == (248, 3)

        pipeline = Pipeline([
            OneHotVectorizer(columns=['age', 'parity', 'education_str']),
            AveragedPerceptronBinaryClassifier(feature='age')
        ])

        pipeline.fit(X, y, verbose=0)

        y_pred_withpipeline = pipeline.predict(X)
        print(y_pred_withpipeline.head())
        assert y_pred_withpipeline.shape == (248, 3)

        metrics, scores = pipeline.test(X, y, output_scores=True)
        print(metrics)
        assert scores.shape == (248, 3)
        assert metrics.shape == (1, 11)

        # back to X_xf1
        print(list(X_xf1.columns))
        l1 = list(sorted(set(_.split('.')[-1] for _ in X_xf1.columns)))
        levels = [['age', 'education', 'education_str', 'parity',
                   'pooled', 'spontaneous', 'stratum', 'induced'], [''] + l1]
        names = ['columns', 'slots']
        labels = [[], []]
        ages = []
        for _ in X_xf1.columns:
            spl = _.split('.')
            l1 = levels[0].index(spl[0])
            try:
                l2 = levels[1].index(spl[1])
            except IndexError:
                l2 = levels[1].index('')
            labels[0].append(l1)
            labels[1].append(l2)
            if spl[0] == 'age':
                ages.append(l2)
        X_xf1.columns = pandas.MultiIndex(
            levels=levels, labels=labels, names=names)
        print(X_xf1.head(n=2).T)

        col_ages = [('age', a) for a in ages]
        print(col_ages)
        try:
            model = AveragedPerceptronBinaryClassifier(feature=col_ages)
            model.fit(X_xf1, y, verbose=0)
            y_pred = model.predict(X_xf1)
            assert y_pred.shape == (248, 3)
        except Exception as e:
            # Does not work, probably confusion between list and tuple in nimbusml
            print(e)

        try:
            model = AveragedPerceptronBinaryClassifier(feature=['age'])
            model.fit(X_xf1, y, verbose=0)
            y_pred = model.predict(X_xf1)
            assert y_pred.shape == (248, 3)
        except Exception as e:
            # Does not work.
            print(e)


if __name__ == "__main__":
    unittest.main()
