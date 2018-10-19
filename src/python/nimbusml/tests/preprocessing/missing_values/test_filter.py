# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------

import os
import unittest

import numpy as np
import pandas as pd
from nimbusml import FileDataStream
from nimbusml.preprocessing.missing_values import Filter


class TestFilter(unittest.TestCase):

    def test_filter(self):
        with_nans = pd.DataFrame(
            data=dict(
                Sepal_Length=[
                    2.5, np.nan, 2.1, 1.0], Sepal_Width=[
                    .75, .9, .8, .76], Petal_Length=[
                    np.nan, 2.5, 2.6, 2.4], Petal_Width=[
                    .8, .7, .9, 0.7]))

        tmpfile = 'tmpfile_with_nans.csv'
        with_nans.to_csv(tmpfile, index=False, na_rep='?')

        file_schema = 'sep=, col=Petal_Length:R4:0 col=Petal_Width:R4:1 ' \
                      'col=Sepal_Length:R4:2 col=Sepal_Width:R4:3 header+'
        data = FileDataStream(tmpfile, schema=file_schema)

        xf = Filter(
            columns=[
                'Petal_Length',
                'Petal_Width',
                'Sepal_Length',
                'Sepal_Width'])

        features = xf.fit_transform(data)

        assert features.shape == (2, 4)
        print(features.columns)
        # columns ordering changed between 0.22 and 0.23
        assert set(
            features.columns) == {
            'Petal_Length',
            'Petal_Width',
            'Sepal_Length',
            'Sepal_Width'}
        os.remove(tmpfile)

    def test_filter_no_renaming(self):
        with_nans = pd.DataFrame(
            data=dict(
                Sepal_Length=[
                    2.5, np.nan, 2.1, 1.0], Sepal_Width=[
                    .75, .9, .8, .76], Petal_Length=[
                    np.nan, 2.5, 2.6, 2.4], Petal_Width=[
                    .8, .7, .9, 0.7], Species=[
                    "setosa", "viginica", "", 'versicolor']))

        tmpfile = 'tmpfile_with_nans.csv'
        with_nans.to_csv(tmpfile, index=False)

        file_schema = 'sep=, col=Petal_Length:R4:0 col=Petal_Width:R4:1 ' \
                      'col=Sepal_Length:R4:2 col=Sepal_Width:R4:3 ' \
                      'col=Species:TX:4 header+'
        data = FileDataStream(tmpfile, schema=file_schema)

        try:
            xf = Filter(columns={'Petal_Length': 'Petal_Length'})
            xf.fit(data)
        except TypeError as e:
            assert 'Dictionaries are not allowed to specify input ' \
                   'columns.' in str(
                       e)

        try:
            xf = Filter(columns={'Petal_Length2': 'Petal_Length'})
            xf.fit(data)
        except TypeError as e:
            assert 'Dictionaries are not allowed to specify input ' \
                   'columns.' in str(
                       e)

    def test_check_estimator_filter(self):
        dataTrain = pd.DataFrame(data=dict(
            Sepal_Length=[2.5, np.nan, 2.1, 1.0],
            Sepal_Width=[.75, .9, .8, .76],
            Petal_Length=[np.nan, 2.5, 2.6, 2.4],
            Petal_Width=[.8, .7, .9, 0.7],
            Species=["setosa", "virginica", "", 'versicolor']))

        filter = Filter() << ["Sepal_Length", "Petal_Length"]
        data_idv = filter.fit_transform(dataTrain)
        assert data_idv is not None
        assert len(data_idv) > 0


if __name__ == '__main__':
    unittest.main()
