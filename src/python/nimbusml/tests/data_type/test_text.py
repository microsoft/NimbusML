# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
import unittest

import numpy as np
import pandas as pd
from nimbusml import Pipeline
from nimbusml.ensemble import LightGbmClassifier
from nimbusml.feature_extraction.text import NGramFeaturizer
from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_array_almost_equal


def transform_data(data=None, datatype=None):
    if datatype == "dataframe":
        return pd.DataFrame(data)
    elif datatype == "array":
        return np.array(data)
    elif datatype == "list":
        return data
    elif datatype == "series":
        return pd.Series(data)


def train_data_type_single(
        fit_X_type="dataframe",
        fit_Y_type=None,
        predict_X_type=None):
    data = [
        "This is sentence 1",
        "Talk about second",
        "Thrid one",
        "Final example."]
    model = NGramFeaturizer()
    data_with_new_type = transform_data(data, fit_X_type)
    model.fit(data_with_new_type)
    test_data_with_new_type = transform_data(data, predict_X_type)
    return model.transform(test_data_with_new_type)


def train_data_type_ppl(fit_X_type=None, fit_Y_type=None, predict_X_type=None):
    data = [
        "This is sentence 1",
        "Talk about second",
        "Thrid one",
        "Final example."]
    label = [1, 0, 1, 1]
    model = Pipeline([
        NGramFeaturizer(),
        LightGbmClassifier(minimum_example_count_per_leaf=1, number_of_threads=1)
    ])
    data_with_new_type = transform_data(data, fit_X_type)
    label_with_new_type = transform_data(label, fit_Y_type)
    model.fit(data_with_new_type, label_with_new_type)
    metrics, scores = model.test(
        data_with_new_type, label_with_new_type, output_scores=True)
    test_data_with_new_type = transform_data(data, predict_X_type)
    return model.predict(test_data_with_new_type), scores, metrics


class TestTextDataType(unittest.TestCase):

    def test_check_text_datatype_single_list_list_series(self):
        result = train_data_type_single("list", "list", "series")
        assert len(result) == 4
        assert len(result.columns) == 66
        assert all([col.startswith('F0') for col in result.columns])

    def test_check_text_datatype_single_series_list_series(self):
        result = train_data_type_single("series", "list", "series")
        assert len(result) == 4
        assert len(result.columns) == 66
        assert all([col.startswith('F0') for col in result.columns])

    def test_check_text_datatype_single_series_list_list(self):
        result = train_data_type_single("series", "list", "list")
        assert len(result) == 4
        assert len(result.columns) == 66
        assert all([col.startswith('F0') for col in result.columns])

    def test_check_text_datatype_single_array_list_series(self):
        result = train_data_type_single("array", "list", "series")
        assert len(result) == 4
        assert len(result.columns) == 66
        assert all([col.startswith('F0') for col in result.columns])

    def test_check_text_datatype_single_series_array_dataframe(self):
        result = train_data_type_single("series", "array", "dataframe")
        assert len(result) == 4
        assert len(result.columns) == 66
        assert all([col.startswith('F0') for col in result.columns])

    def test_check_text_datatype_single_array_series_series(self):
        result = train_data_type_single("array", "series", "series")
        assert len(result) == 4
        assert len(result.columns) == 66
        assert all([col.startswith('F0') for col in result.columns])

    def test_check_text_datatype_single_dataframe_list_series(self):
        result = train_data_type_single("dataframe", "list", "series")
        assert len(result) == 4
        assert len(result.columns) == 66
        assert all([col.startswith('F0') for col in result.columns])

    def test_check_text_datatype_single_series_series_dataframe(self):
        result = train_data_type_single("series", "series", "dataframe")
        assert len(result) == 4
        assert len(result.columns) == 66
        assert all([col.startswith('F0') for col in result.columns])

    def test_check_text_datatype_single_dataframe_series_list(self):
        result = train_data_type_single("dataframe", "series", "list")
        assert len(result) == 4
        assert len(result.columns) == 66
        assert all([col.startswith('F0') for col in result.columns])

    def test_check_text_datatype_ppl_series_list_array(self):
        result, scores, metrics = train_data_type_ppl(
            "series", "list", "array")
        assert len(result) == 4
        assert_almost_equal(metrics['Log-loss'].item(), 0.56233514)
        assert_array_equal(scores['Score.0'].values, result['Score.0'].values)
        assert_array_almost_equal(scores['Score.0'].values, [0.25, 0.25, 0.25, 0.25])

    def test_check_text_datatype_ppl_list_series_dataframe(self):
        result, scores, metrics = train_data_type_ppl(
            "list", "series", "dataframe")
        assert len(result) == 4
        assert_almost_equal(metrics['Log-loss'].item(), 0.56233514)
        assert_array_equal(scores['Score.0'].values, result['Score.0'].values)
        assert_array_almost_equal(scores['Score.0'].values, [0.25, 0.25, 0.25, 0.25])

    def test_check_text_datatype_ppl_list_list_series(self):
        result, scores, metrics = train_data_type_ppl("list", "list", "series")
        assert len(result) == 4
        assert_almost_equal(metrics['Log-loss'].item(), 0.56233514)
        assert_array_equal(scores['Score.0'].values, result['Score.0'].values)
        assert_array_almost_equal(scores['Score.0'].values, [0.25, 0.25, 0.25, 0.25])

    def test_check_text_datatype_ppl_array_series_array(self):
        result, scores, metrics = train_data_type_ppl(
            "array", "series", "array")
        assert len(result) == 4
        assert_almost_equal(metrics['Log-loss'].item(), 0.56233514)
        assert_array_equal(scores['Score.0'].values, result['Score.0'].values)
        assert_array_almost_equal(scores['Score.0'].values, [0.25, 0.25, 0.25, 0.25])

    def test_check_text_datatype_ppl_series_array_dataframe(self):
        result, scores, metrics = train_data_type_ppl(
            "series", "array", "dataframe")
        assert len(result) == 4
        assert_almost_equal(metrics['Log-loss'].item(), 0.56233514)
        assert_array_equal(scores['Score.0'].values, result['Score.0'].values)
        assert_array_almost_equal(scores['Score.0'].values, [0.25, 0.25, 0.25, 0.25])

    def test_check_text_datatype_ppl_array_series_list(self):
        result, scores, metrics = train_data_type_ppl(
            "array", "series", "list")
        assert len(result) == 4
        assert_almost_equal(metrics['Log-loss'].item(), 0.56233514)
        assert_array_equal(scores['Score.0'].values, result['Score.0'].values)
        assert_array_almost_equal(scores['Score.0'].values, [0.25, 0.25, 0.25, 0.25])

    def test_check_text_datatype_ppl_dataframe_list_series(self):
        result, scores, metrics = train_data_type_ppl(
            "dataframe", "list", "series")
        assert len(result) == 4
        assert_almost_equal(metrics['Log-loss'].item(), 0.56233514)
        assert_array_equal(scores['Score.0'].values, result['Score.0'].values)
        assert_array_almost_equal(scores['Score.0'].values, [0.25, 0.25, 0.25, 0.25])

    def test_check_text_datatype_ppl_series_series_dataframe(self):
        result, scores, metrics = train_data_type_ppl(
            "series", "series", "dataframe")
        assert len(result) == 4
        assert_almost_equal(metrics['Log-loss'].item(), 0.56233514)
        assert_array_equal(scores['Score.0'].values, result['Score.0'].values)
        assert_array_almost_equal(scores['Score.0'].values, [0.25, 0.25, 0.25, 0.25])

    def test_check_text_datatype_ppl_dataframe_series_series(self):
        result, scores, metrics = train_data_type_ppl(
            "dataframe", "series", "series")
        assert len(result) == 4
        assert_almost_equal(metrics['Log-loss'].item(), 0.56233514)
        assert_array_equal(scores['Score.0'].values, result['Score.0'].values)
        assert_array_almost_equal(scores['Score.0'].values, [0.25, 0.25, 0.25, 0.25])


if __name__ == '__main__':
    unittest.main()
