# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import unittest

import numpy
import pandas
from nimbusml import Pipeline
from nimbusml.preprocessing import FromKey, ToKey
from nimbusml.preprocessing.schema import ColumnConcatenator
from pandas import Categorical
from sklearn.utils.testing import assert_raise_message


class TestFromKey(unittest.TestCase):

    def test_example_key_to_text_typeerror_i8(self):
        text_df = pandas.DataFrame(data=dict(text=[1, 2]))
        tokey = FromKey() << 'text'
        # System.ArgumentOutOfRangeException: 'Source column 'text' has invalid
        # type ('I8'): Expected Key type of known cardinality.
        assert_raise_message(
            RuntimeError,
            "",
            lambda: tokey.fit_transform(text_df))

    def test_example_key_to_text_typeerror_u4(self):
        text_df = pandas.DataFrame(data=dict(text=[1, 2]), dtype=numpy.uint32)
        tokey = FromKey() << 'text'
        # System.ArgumentOutOfRangeException: 'Source column 'text' has invalid
        # type ('U8'): Expected Key type of known cardinality.
        assert_raise_message(
            RuntimeError,
            "",
            lambda: tokey.fit_transform(text_df))

    def test_check_estimator_fromkey(self):
        text_df = pandas.DataFrame(
            data=dict(
                text=[
                    "cat",
                    "dog",
                    "fish",
                    "orange",
                    "cat orange",
                    "dog",
                    "fish",
                    "spider"],
                num=[
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8]))

        tokey = ToKey() << ['text']
        data_idv = tokey.fit_transform(text_df)
        assert data_idv is not None
        assert len(data_idv) > 0
        assert str(sorted([str(dt) for dt in data_idv.dtypes])
                   ) == "['category', 'int64']"
        fromkey = FromKey() << ['text']
        data = fromkey.fit_transform(data_idv)
        assert str(list(data_idv['text'])) == str(list(data['text']))
        t = numpy.unique(data_idv['text'].cat.codes)
        assert len(t) == 6
        assert list(data_idv['text'].cat.categories) == [
            "cat", "dog", "fish", "orange", "cat orange", "spider"]

    def test_check_estimator_fromkey_categories(self):
        text_df = pandas.DataFrame(
            data=dict(
                text=[
                    "cat",
                    "dog",
                    "fish",
                    "orange",
                    "cat orange",
                    "dog",
                    "fish",
                    "spider"]),
            dtype="category")

        tokey = ToKey() << ['text']
        data_idv = tokey.fit_transform(text_df)
        assert data_idv is not None
        assert len(data_idv) > 0
        assert data_idv['text'].dtype == 'category'

    def test_fromkey_multiple_columns(self):
        df = pandas.DataFrame(data=dict(
            num1=[0, 1, 2, 3, 4, 5, 6],
            cat1=Categorical.from_codes([0, 2, 3, 1, 2, -1, 1],
                                        categories=["a", "b", "c", "d"]),
            cat2=Categorical.from_codes([2, 0, 1, 2, 0, 1, 1],
                                        categories=["e", "f", "g"]),
            num=[0, 1, 2, 3, 4, 5, 6],
            text1=["i", "j", "i", "j", "i", "j", "i"],
            text2=["k", "l", "l", "k", "k", "l", "k"]))

        concat = ColumnConcatenator() << {'textvec': ['text1', 'text2']}
        tokey = ToKey() << ['textvec']
        pipeline = Pipeline([concat, tokey])
        data_idv = pipeline.fit_transform(df)
        assert sorted(
            list(
                data_idv.columns)) == [
            'cat1',
            'cat2',
            'num',
            'num1',
            'text1',
            'text2',
            'textvec.text1',
            'textvec.text2']
        assert list(data_idv['cat1'].cat.categories) == ['a', 'b', 'c', 'd']
        assert list(data_idv['cat1'].cat.codes) == [0, 2, 3, 1, 2, -1, 1]
        assert list(data_idv['cat2'].cat.categories) == ['e', 'f', 'g']
        assert list(data_idv['cat2'].cat.codes) == [2, 0, 1, 2, 0, 1, 1]
        assert list(
            data_idv['textvec.text1'].cat.categories) == [
            'i', 'k', 'j', 'l']
        assert list(data_idv['textvec.text1'].cat.codes) == [
            0, 2, 0, 2, 0, 2, 0]
        assert list(
            data_idv['textvec.text2'].cat.categories) == [
            'i', 'k', 'j', 'l']
        assert list(data_idv['textvec.text2'].cat.codes) == [
            1, 3, 3, 1, 1, 3, 1]


if __name__ == '__main__':
    unittest.main()
