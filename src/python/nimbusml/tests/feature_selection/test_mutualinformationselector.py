# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import unittest

import numpy
import pandas
from nimbusml import Pipeline, Role
from nimbusml.feature_selection import MutualInformationSelector


class TestMutualInformationSelector(unittest.TestCase):

    def test_example_success(self):

        like = [
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
            True]
        x1 = [(5. if _ else 4.) for _ in like]
        x2 = [(-5. if _ else -4.) for _ in like]
        x1[0] = 50
        x2[1] = 50
        x2[2] = 50
        train_data = pandas.DataFrame(
            data=dict(
                like=like,
                x1=x2,
                x2=x2),
            dtype=numpy.float32)

        X = train_data.drop('like', axis=1)
        y = train_data[['like']]
        transform_2 = MutualInformationSelector()
        exp = Pipeline([transform_2])
        res = exp.fit_transform(X, y)
        assert res is not None

        transform_2 = MutualInformationSelector(slots_in_output=2)
        pipe = Pipeline([transform_2])
        res = pipe.fit_transform(X, y)
        assert res is not None

        transform_2 = MutualInformationSelector() << {
            Role.Feature: [
                'x1', 'x2'], Role.Label: 'like'}
        assert transform_2._allowed_roles == {'Label'}
        assert transform_2.label_column_name == 'like'
        assert transform_2.input == ['x1', 'x2']
        assert transform_2.output == ['Feature']
        exp = Pipeline([transform_2])
        res = exp.fit_transform(train_data)
        assert res is not None

        transform_2 = MutualInformationSelector(
        ) << {"zoo": ['x1', 'x2'], Role.Label: 'like'}
        assert transform_2._allowed_roles == {'Label'}
        assert transform_2.label_column_name == 'like'
        assert transform_2.input == ['x1', 'x2']
        assert transform_2.output == ['zoo']
        exp = Pipeline([transform_2])
        res = exp.fit_transform(train_data)
        assert res is not None

        transform_2 = MutualInformationSelector() << {
            "zoo": ['x1'], Role.Label: 'like'}
        assert transform_2._allowed_roles == {'Label'}
        assert transform_2.label_column_name == 'like'
        assert transform_2.input == ['x1']
        assert transform_2.output == ['zoo']
        exp = Pipeline([transform_2])
        res = exp.fit_transform(train_data)
        assert res is not None

        transform_2 = MutualInformationSelector(
            slots_in_output=1, columns=['x1'], label='like')
        assert transform_2._allowed_roles == {'Label'}
        assert transform_2.label_column_name == 'like'
        assert transform_2.input == ['x1']
        assert transform_2.output == ['x1']
        pipe = Pipeline([transform_2])
        pipe.fit(train_data)
        res = pipe.transform(train_data)
        assert res is not None

    def test_example_fails(self):

        like = [
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
            True]
        x1 = [(5. if _ else 4.) for _ in like]
        x2 = [(-5. if _ else -4.) for _ in like]
        x1[0] = 50
        x2[1] = 50
        x2[2] = 50
        train_data = pandas.DataFrame(
            data=dict(
                like=like,
                x1=x2,
                x2=x2),
            dtype=numpy.float32)

        # It works but I'm not sure what it does.
        transform_2 = MutualInformationSelector(
            slots_in_output=1, feature=[
                'x1', 'x2'], label='like')
        assert transform_2._allowed_roles == {'Label'}
        assert transform_2.label_column_name == 'like'
        # assert transform_2.input == ['x1', 'x2']  # None
        # assert transform_2.output == ['Feature'] # None
        pipe = Pipeline([transform_2])
        pipe.fit(train_data)
        res = pipe.transform(train_data)
        assert res is not None

        # It works but I'm not sure what it does.
        try:
            transform_2 = MutualInformationSelector(
                slots_in_output=1, feature2=[
                    'x1', 'x2'], label='like')
            raise AssertionError("feature2 not allowed")
        except NameError as e:
            assert "Parameter 'feature2' is not allowed" in str(e)

        try:
            transform_2 = MutualInformationSelector(
                slots_in_output=2, columns=['x1', 'x2'], label='like')
            raise AssertionError("only one output is allowed")
        except RuntimeError as e:
            assert "use a dictionary" in str(e)

        try:
            transform_2 = MutualInformationSelector(
                slots_in_output=2, columns={
                    'x1': 'x1', 'x2': 'x2'}, label='like')
            raise AssertionError("only one output is allowed")
        except RuntimeError as e:
            assert "Output should contain only one output not" in str(e)


if __name__ == '__main__':
    unittest.main()
