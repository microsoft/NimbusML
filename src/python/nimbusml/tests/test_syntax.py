# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
import unittest

import pandas
import six
from nimbusml import Pipeline
from nimbusml.feature_extraction.categorical import OneHotVectorizer, \
    OneHotHashVectorizer
from nimbusml.feature_extraction.text import NGramFeaturizer
from nimbusml.feature_selection import MutualInformationSelector
from nimbusml.internal.entrypoints._ngramextractor_ngram import n_gram
from nimbusml.internal.utils.data_roles import Role
from nimbusml.linear_model import FastLinearBinaryClassifier
from nimbusml.preprocessing.normalization import LogMeanVarianceScaler
from nimbusml.preprocessing.schema import ColumnConcatenator as Concat, \
    ColumnDropper as Drop
# from sklearn.pipeline import Pipeline

if six.PY2:
    pass
else:
    pass


class TestSyntax(unittest.TestCase):

    def test_syntax1(self):

        df = pandas.DataFrame(dict(education=['A', 'B', 'A', 'B', 'A'],
                                   workclass=['X', 'X', 'Y', 'Y', 'Y'],
                                   y=[1, 0, 1, 0, 0]))
        X = df.drop('y', axis=1)
        y = df['y']

        exp = Pipeline([
            OneHotVectorizer(),
            FastLinearBinaryClassifier(maximum_number_of_iterations=1)
        ])
        exp.fit(X, y)
        prediction = exp.predict(X)
        assert isinstance(prediction, pandas.DataFrame)
        assert sorted(list(prediction.columns)) == [
            'PredictedLabel', 'Probability', 'Score']
        assert prediction.shape == (5, 3)

    def test_syntax2(self):

        df = pandas.DataFrame(dict(education=['A', 'B', 'A', 'B', 'A'],
                                   workclass=['X', 'X', 'Y', 'Y', 'Y'],
                                   y=[1, 0, 1, 0, 0]))
        X = df.drop('y', axis=1)
        y = df['y']

        exp = Pipeline([
            OneHotVectorizer() << 'education',
            OneHotVectorizer(max_num_terms=2) << 'workclass',
            FastLinearBinaryClassifier(maximum_number_of_iterations=1)
        ])
        exp.fit(X, y)
        prediction = exp.predict(X)
        assert isinstance(prediction, pandas.DataFrame)
        assert sorted(list(prediction.columns)) == [
            'PredictedLabel', 'Probability', 'Score']
        assert prediction.shape == (5, 3)

    def test_syntax2_lt(self):

        df = pandas.DataFrame(dict(education=['A', 'B', 'A', 'B', 'A'],
                                   workclass=['X', 'X', 'Y', 'Y', 'Y'],
                                   y=[1, 0, 1, 0, 0]))
        X = df.drop('y', axis=1)
        y = df['y']

        exp = Pipeline([
            OneHotVectorizer() << 'education',
            OneHotVectorizer(max_num_terms=2) << 'workclass',
            FastLinearBinaryClassifier(maximum_number_of_iterations=1)
        ])
        exp.fit(X, y)
        prediction = exp.predict(X)
        assert isinstance(prediction, pandas.DataFrame)
        assert sorted(list(prediction.columns)) == [
            'PredictedLabel', 'Probability', 'Score']
        assert prediction.shape == (5, 3)

    def test_syntax3(self):

        df = pandas.DataFrame(dict(education=['A', 'B', 'A', 'B', 'A'],
                                   workclass=['X', 'X', 'Y', 'Y', 'Y'],
                                   y=[1, 0, 1, 0, 0]))
        X = df.drop('y', axis=1)
        y = df['y']

        exp = Pipeline([
            OneHotVectorizer() << {'edu1': 'education'},
            OneHotHashVectorizer() << 'education',
            OneHotVectorizer(max_num_terms=2) << 'workclass',
            # Currently the learner does not use edu1
            # unless it is specified explicitely so nimbusml
            # does not do what the syntax implicetely tells.
            # We need to modify either the bridge to look into
            # every available column at one step.
            FastLinearBinaryClassifier(maximum_number_of_iterations=1)
        ])
        exp.fit(X, y)
        prediction = exp.predict(X)
        assert isinstance(prediction, pandas.DataFrame)
        assert sorted(list(prediction.columns)) == [
            'PredictedLabel', 'Probability', 'Score']
        assert prediction.shape == (5, 3)

    def test_syntax4(self):

        df = pandas.DataFrame(dict(education=['A', 'B', 'A', 'B', 'A'],
                                   workclass=['X', 'X', 'Y', 'Y', 'Y'],
                                   y=[1, 0, 1, 0, 0]))
        X = df.drop('y', axis=1)
        y = df['y']

        exp = Pipeline([
            OneHotVectorizer() << {'edu1': 'education'},
            OneHotHashVectorizer() << {'edu2': 'education'},
            OneHotVectorizer(max_num_terms=2) << {'wki': 'workclass'},
            Concat() << {'Inputs': ['edu1', 'edu2', 'wki']},
            FastLinearBinaryClassifier(maximum_number_of_iterations=1) << 'Inputs'
        ])
        exp.fit(X, y)
        prediction = exp.predict(X)
        assert isinstance(prediction, pandas.DataFrame)
        assert sorted(list(prediction.columns)) == [
            'PredictedLabel', 'Probability', 'Score']
        assert prediction.shape == (5, 3)

    def test_syntax4_2(self):

        df = pandas.DataFrame(dict(education=['A', 'B', 'A', 'B', 'A'],
                                   workclass=['X', 'X', 'Y', 'Y', 'Y'],
                                   y=[1, 0, 1, 0, 0]))
        X = df.drop('y', axis=1)
        y = df['y']

        exp = Pipeline([
            OneHotVectorizer() << {'edu1': 'education'},
            OneHotHashVectorizer() << {'edu2': 'education'},
            OneHotVectorizer(max_num_terms=2) << {'wki': 'workclass'},
            Concat() << {'Inputs': ['edu1', 'edu2', 'wki']},
            FastLinearBinaryClassifier(maximum_number_of_iterations=1) << 'Inputs'
        ])
        exp.fit(X, y)
        prediction = exp.predict(X)
        assert isinstance(prediction, pandas.DataFrame)
        assert sorted(list(prediction.columns)) == [
            'PredictedLabel', 'Probability', 'Score']
        assert prediction.shape == (5, 3)

    def test_syntax4_dict(self):

        df = pandas.DataFrame(dict(education=['A', 'B', 'A', 'B', 'A'],
                                   workclass=['X', 'X', 'Y', 'Y', 'Y'],
                                   y=[1, 0, 1, 0, 0]))
        X = df.drop('y', axis=1)
        y = df['y']

        exp = Pipeline([
            OneHotVectorizer() << {'edu1': 'education'},
            OneHotHashVectorizer() << {'edu2': 'education'},
            OneHotVectorizer(max_num_terms=2) << {'wki': 'workclass'},
            Concat() << {'Inputs': ['edu1', 'edu2', 'wki']},
            FastLinearBinaryClassifier(maximum_number_of_iterations=1) << 'Inputs'
        ])
        exp.fit(X, y)
        prediction = exp.predict(X)
        assert isinstance(prediction, pandas.DataFrame)
        assert sorted(list(prediction.columns)) == [
            'PredictedLabel', 'Probability', 'Score']
        assert prediction.shape == (5, 3)

    def test_syntax4_columns(self):

        df = pandas.DataFrame(dict(education=['A', 'B', 'A', 'B', 'A'],
                                   workclass=['X', 'X', 'Y', 'Y', 'Y'],
                                   y=[1, 0, 1, 0, 0]))
        X = df.drop('y', axis=1)
        y = df['y']

        exp = Pipeline([
            OneHotVectorizer(columns={'edu1': 'education'}),
            OneHotHashVectorizer(columns={'edu2': 'education'}),
            OneHotVectorizer(max_num_terms=2, columns={'wki': 'workclass'}),
            Concat(columns={'Inputs': ['edu1', 'edu2', 'wki']}),
            FastLinearBinaryClassifier(maximum_number_of_iterations=1) << 'Inputs'
        ])
        exp.fit(X, y)
        prediction = exp.predict(X)
        assert isinstance(prediction, pandas.DataFrame)
        assert sorted(list(prediction.columns)) == [
            'PredictedLabel', 'Probability', 'Score']
        assert prediction.shape == (5, 3)

    @unittest.skip(
        "skip until we have a proper way to catch exception raised by nimbusml")
    def test_syntax4_fail(self):

        df = pandas.DataFrame(dict(education=['A', 'B', 'A', 'B', 'A'],
                                   workclass=['X', 'X', 'Y', 'Y', 'Y'],
                                   y=[1, 0, 1, 0, 0]))
        X = df.drop('y', axis=1)
        y = df['y']

        exp = Pipeline([
            OneHotVectorizer() << {'edu1': 'education'},
            OneHotHashVectorizer() << {'edu2': 'education'},
            OneHotVectorizer(max_num_terms=2) << {'wki': 'workclass'},
            FastLinearBinaryClassifier(maximum_number_of_iterations=1) << ['edu1', 'edu2',
                                                             'wki']
        ])
        try:
            exp.fit(X, y)
            assert False
        except RuntimeError as e:
            assert "ConcatTransform() << {'Input': ['edu1', 'edu2', 'wki']}" \
                   in str(e)

    @unittest.skip(
        "skip until we have a proper way to catch exception raised by nimbusml")
    def test_syntax4_fail2(self):

        df = pandas.DataFrame(dict(education=['A', 'B', 'A', 'B', 'A'],
                                   workclass=['X', 'X', 'Y', 'Y', 'Y'],
                                   y=[1, 0, 1, 0, 0]))
        X = df.drop('y', axis=1)
        y = df['y']

        exp = Pipeline([
            OneHotVectorizer() << {'edu1': 'education'},
            OneHotHashVectorizer() << {'edu2': 'education'},
            OneHotVectorizer(max_num_terms=2) << {'wki': 'workclass'},
            FastLinearBinaryClassifier(maximum_number_of_iterations=1) << ['edu1', 'edu4',
                                                             'wki']
        ])
        try:
            exp.fit(X, y)
            raise AssertionError("The test should not reach this line.")
        except Exception as e:
            assert "Feature column 'edu4' not found" in str(e)

    def test_syntax5(self):
        df = pandas.DataFrame(dict(education=['A', 'B', 'A', 'B', 'A'],
                                   workclass=['X', 'X', 'Y', 'Y', 'Y'],
                                   y=[1, 0, 1, 0, 0]))
        X = df.drop('y', axis=1)
        y = df['y']

        exp = Pipeline([
            OneHotVectorizer() << {'f1': 'education'},
            OneHotHashVectorizer() << {'f2': 'education'},
            OneHotVectorizer(max_num_terms=2) << {'f3': 'workclass'},
            Concat() << {'Features': ['f%d' % i for i in range(1, 4)]},
            FastLinearBinaryClassifier(maximum_number_of_iterations=1) << 'Features'
        ])
        exp.fit(X, y)
        prediction = exp.predict(X)
        assert isinstance(prediction, pandas.DataFrame)
        assert sorted(list(prediction.columns)) == [
            'PredictedLabel', 'Probability', 'Score']
        assert prediction.shape == (5, 3)

    @unittest.skip("regular expression not yet implemented")
    def test_syntax5_regular_expression(self):
        # REVIEW: not implemented yet
        # The best would be to handle regular expression inside nimbusml.
        # It could be handled in entrypoint.py just before calling nimbusml.
        # It can be handled inside Pipeline if it is aware of
        # the input schema.

        df = pandas.DataFrame(dict(education=['A', 'B', 'A', 'B', 'A'],
                                   workclass=['X', 'X', 'Y', 'Y', 'Y'],
                                   y=[1, 0, 1, 0, 0]))
        X = df.drop('y', axis=1)
        y = df['y']

        exp = Pipeline([
            OneHotVectorizer() << {'f1': 'education'},
            OneHotHashVectorizer() << {'f2': 'education'},
            OneHotVectorizer(max_num_terms=2) << {'f3': 'workclass'},
            Concat() << {'Features': 'f[0-9]+'},
            FastLinearBinaryClassifier(maximum_number_of_iterations=1) << 'Features'
        ])
        exp.fit(X, y)
        prediction = exp.predict(X)
        assert isinstance(prediction, pandas.DataFrame)
        assert sorted(list(prediction.columns)) == [
            'PredictedLabel', 'Probability', 'Score']
        assert prediction.shape == (5, 3)

    def test_syntax6(self):

        df = pandas.DataFrame(dict(education=['A', 'B', 'A', 'B', 'A'],
                                   workclass=['X', 'X', 'Y', 'Y', 'Y'],
                                   y=[1, 0, 1, 0, 0]))
        X = df.drop('y', axis=1)
        y = df['y']

        exp = Pipeline([
            OneHotVectorizer() << {'f1': 'education'},
            OneHotHashVectorizer() << {'f2': 'education'},
            OneHotVectorizer(max_num_terms=2) << {'f3': 'workclass'},
            Concat() << {'Features': ['f%d' % i for i in range(1, 4)]},
            Drop() << ['education', 'workclass', 'f1', 'f2', 'f3'],
            FastLinearBinaryClassifier(maximum_number_of_iterations=1) << ['Features']
        ])
        exp.fit(X, y)
        prediction = exp.predict(X)
        assert isinstance(prediction, pandas.DataFrame)
        assert sorted(list(prediction.columns)) == [
            'PredictedLabel', 'Probability', 'Score']
        assert prediction.shape == (5, 3)

    def test_syntax6_not_features(self):

        df = pandas.DataFrame(dict(education=['A', 'B', 'A', 'B', 'A'],
                                   workclass=['X', 'X', 'Y', 'Y', 'Y'],
                                   y=[1, 0, 1, 0, 0]))
        X = df.drop('y', axis=1)
        y = df['y']

        exp = Pipeline([
            OneHotVectorizer() << {'f1': 'education'},
            OneHotHashVectorizer() << {'f2': 'education'},
            OneHotVectorizer(max_num_terms=2) << {'f3': 'workclass'},
            Concat() << {'FeaturesCustom': ['f%d' % i for i in range(1, 4)]},
            Drop() << ['education', 'workclass', 'f1', 'f2', 'f3'],
            FastLinearBinaryClassifier(maximum_number_of_iterations=1) << 'FeaturesCustom'
        ])
        exp.fit(X, y)
        prediction = exp.predict(X)
        assert isinstance(prediction, pandas.DataFrame)
        assert sorted(list(prediction.columns)) == [
            'PredictedLabel', 'Probability', 'Score']
        assert prediction.shape == (5, 3)

    @unittest.skip(reason="what should be the default behavior")
    def test_syntax6_change_role(self):
        # REVIEW: the pipeline drops all columns but one -->
        # nimbusml still thinks the Features are eduction, workclass
        # and does not automatically detects that the only remaining
        # columns should play that role
        # (maybe because the label column is here too even though
        # the only remaining column without a role is Features).
        df = pandas.DataFrame(dict(education=['A', 'B', 'A', 'B', 'A'],
                                   workclass=['X', 'X', 'Y', 'Y', 'Y'],
                                   y=[1, 0, 1, 0, 0]))
        X = df.drop('y', axis=1)
        y = df['y']

        exp = Pipeline([
            OneHotVectorizer() << {'f1': 'education'},
            OneHotHashVectorizer() << {'f2': 'education'},
            OneHotVectorizer(max_num_terms=2) << {'f3': 'workclass'},
            Concat() << {'Features': ['f%d' % i for i in range(1, 4)]},
            Drop() << ['education', 'workclass', 'f1', 'f2', 'f3'],
            FastLinearBinaryClassifier(maximum_number_of_iterations=1) << ['Features']
        ])
        exp.fit(X, y)
        prediction = exp.predict(X)
        assert isinstance(prediction, pandas.DataFrame)
        assert sorted(list(prediction.columns)) == [
            'PredictedLabel', 'Probability', 'Score']
        assert prediction.shape == (5, 3)

    @unittest.skip("regular expression not yet implemented")
    def test_syntax6_regular_expression(self):

        df = pandas.DataFrame(dict(education=['A', 'B', 'A', 'B', 'A'],
                                   workclass=['X', 'X', 'Y', 'Y', 'Y'],
                                   y=[1, 0, 1, 0, 0]))
        X = df.drop('y', axis=1)
        y = df['y']

        exp = Pipeline([
            OneHotVectorizer() << {'f1': 'education'},
            OneHotHashVectorizer() << {'f2': 'education'},
            OneHotVectorizer(max_num_terms=2) << {'f3': 'workclass'},
            Concat() << {'Features': ['f%d' % i for i in range(1, 4)]},
            Drop() << '~Features',
            FastLinearBinaryClassifier(maximum_number_of_iterations=1)
        ])
        exp.fit(X, y)
        prediction = exp.predict(X)
        assert isinstance(prediction, pandas.DataFrame)
        assert sorted(list(prediction.columns)) == [
            'PredictedLabel', 'Probability', 'Score']
        assert prediction.shape == (5, 3)

    def test_syntax9_slots_label(self):

        train_reviews = pandas.DataFrame(
            data=dict(
                review=[
                    "This is great",
                    "I hate it",
                    "Love it",
                    "Do not like it",
                    "Really like it",
                    "I hate it",
                    "I like it a lot",
                    "I kind of hate it",
                    "I do like it",
                    "I really hate it",
                    "It is very good",
                    "I hate it a bunch",
                    "I love it a bunch",
                    "I hate it",
                    "I like it very much",
                    "I hate it very much.",
                    "I really do love it",
                    "I really do hate it",
                    "Love it!",
                    "Hate it!",
                    "I love it",
                    "I hate it",
                    "I love it",
                    "I hate it",
                    "I love it"],
                like=[
                    True,
                    False,
                    True,
                    False,
                    True,
                    False,
                    True,
                    False,
                    True,
                    False,
                    True,
                    False,
                    True,
                    False,
                    True,
                    False,
                    True,
                    False,
                    True,
                    False,
                    True,
                    False,
                    True,
                    False,
                    True]))

        X = train_reviews.loc[:, train_reviews.columns != 'like']
        y = train_reviews[['like']]

        transform_1 = NGramFeaturizer(word_feature_extractor=n_gram())
        transform_2 = MutualInformationSelector()
        exp = Pipeline([transform_1, transform_2])
        res = exp.fit_transform(X, y)
        assert res is not None

        # Scikit compatibility (Compose transforms inside Scikit Pipeline).
        # In this scenario, we do not provide {input, output} arguments
        transform_1 = NGramFeaturizer(word_feature_extractor=n_gram())
        transform_2 = MutualInformationSelector(slots_in_output=2)
        pipe = Pipeline([transform_1, transform_2])
        res = pipe.fit_transform(X, y)
        assert res is not None

    def test_syntax10_multi_output1(self):
        in_df = pandas.DataFrame(
            data=dict(
                Sepal_Length=[
                    2.5, 1, 2.1, 1.0], Sepal_Width=[
                    .75, .9, .8, .76], Petal_Length=[
                    0, 2.5, 2.6, 2.4], Species=[
                    "setosa", "viginica", "setosa", 'versicolor']))

        # generate two new Columns - Petal_Normed and Sepal_Normed
        normed = LogMeanVarianceScaler() << {
            'Petal_Normed': 'Petal_Length',
            'Sepal_Normed': 'Sepal_Width'}
        out_df = normed.fit_transform(in_df)
        self.assertEqual(sorted(list(out_df.columns)),
                         ['Petal_Length', 'Petal_Normed', 'Sepal_Length',
                          'Sepal_Normed', 'Sepal_Width', 'Species'])

    def test_syntax10_multi_output2(self):
        in_df = pandas.DataFrame(
            data=dict(
                Sepal_Length=[
                    2.5, 1, 2.1, 1.0], Sepal_Width=[
                    .75, .9, .8, .76], Petal_Length=[
                    0, 2.5, 2.6, 2.4], Species=[
                    "setosa", "viginica", "setosa", 'versicolor']))

        # generate two new Columns - Petal_Normed and Sepal_Normed
        newcols = {
            'Petal_Normed': 'Petal_Length',
            'Sepal_Normed': 'Sepal_Width'}
        normed = LogMeanVarianceScaler() << newcols
        out_df = normed.fit_transform(in_df)
        self.assertEqual(sorted(list(out_df.columns)),
                         ['Petal_Length', 'Petal_Normed', 'Sepal_Length',
                          'Sepal_Normed', 'Sepal_Width', 'Species'])

    def test_syntax11_learner(self):

        df = pandas.DataFrame(dict(education=['A', 'B', 'A', 'B', 'A'],
                                   workclass=['X', 'X', 'Y', 'Y', 'Y'],
                                   y=[1, 0, 1, 0, 0]))
        X = df.drop('y', axis=1)

        exp = Pipeline(
            [
                OneHotVectorizer() << {
                    'edu1': 'education'}, OneHotHashVectorizer() << {
                    'edu2': 'education'}, FastLinearBinaryClassifier(
                    maximum_number_of_iterations=1) << {
                        'Features': ['edu1', 'edu2'], Role.Label: 'y'}])
        exp.fit(df)
        prediction = exp.predict(X)
        assert isinstance(prediction, pandas.DataFrame)
        assert sorted(list(prediction.columns)) == [
            'PredictedLabel', 'Probability', 'Score']
        assert prediction.shape == (5, 3)

    def test_syntax11_append_insert(self):

        df = pandas.DataFrame(dict(education=['A', 'B', 'A', 'B', 'A'],
                                   workclass=['X', 'X', 'Y', 'Y', 'Y'],
                                   y=[1, 0, 1, 0, 0]))
        X = df.drop('y', axis=1)

        exp = Pipeline()
        exp.append(
            ("OneHotHashVectorizer",
             OneHotHashVectorizer() << {
                 'edu2': 'education'}))
        exp.insert(0, OneHotVectorizer() << {'edu1': 'education'})
        exp.append(
            FastLinearBinaryClassifier(
                maximum_number_of_iterations=1) << {
                'Features': [
                    'edu1',
                    'edu2'],
                Role.Label: 'y'})
        exp.append(OneHotHashVectorizer() << {'edu2': 'education'})
        del exp[-1]
        assert len(exp) == 3

        exp.fit(df)
        prediction = exp.predict(X)
        assert isinstance(prediction, pandas.DataFrame)
        assert sorted(list(prediction.columns)) == [
            'PredictedLabel', 'Probability', 'Score']
        assert prediction.shape == (5, 3)

        try:
            exp.append(OneHotHashVectorizer() << {'edu2': 'education'})
        except RuntimeError as e:
            assert "Model is fitted and cannot be modified" in str(e)
        try:
            exp.insert(0, OneHotHashVectorizer() << {'edu2': 'education'})
        except RuntimeError as e:
            assert "Model is fitted and cannot be modified" in str(e)
        try:
            del exp[0]
        except RuntimeError as e:
            assert "Model is fitted and cannot be modified" in str(e)

        obj = exp[1][1]
        assert obj.__class__.__name__ == "OneHotHashVectorizer"
        obj = exp[1][1]
        assert obj.__class__.__name__ == "OneHotHashVectorizer"
        res = exp['OneHotHashVectorizer']
        assert len(res) == 1
        graph = exp.graph_
        assert len(graph.nodes) >= len(exp)


if __name__ == "__main__":
    unittest.main()
