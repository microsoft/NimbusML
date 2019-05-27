# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
import unittest

import numpy as np
import pandas as pd
from nimbusml import Pipeline
from nimbusml.ensemble import LightGbmClassifier
from nimbusml.linear_model import LogisticRegressionBinaryClassifier
from nimbusml.preprocessing.normalization import Binner, MinMaxScaler
from scipy.sparse import csr_matrix
from sklearn.utils.testing import assert_array_equal


def transform_data(data=None, datatype=None):
    if datatype == "dataframe":
        return pd.DataFrame(data)
    elif datatype == "array":
        return np.array(data)
    elif datatype == "list":
        return data
    elif datatype == "sparse":
        return csr_matrix(data)


def train_data_type_single(
        fit_X_type="dataframe",
        fit_Y_type=None,
        predict_X_type=None):
    data = [[1, 2, 3], [2, 3, 4], [1, 2, 3], [2, 2, 2]]
    label = [1, 0, 1, 1]
    if fit_X_type == "sparse":
        model = LightGbmClassifier(minimum_example_count_per_leaf=1)
    else:
        model = LogisticRegressionBinaryClassifier()
    data_with_new_type = transform_data(data, fit_X_type)
    label_with_new_type = transform_data(label, fit_Y_type)
    model.fit(data_with_new_type, label_with_new_type)
    test_data_with_new_type = transform_data(data, predict_X_type)
    return model.predict(test_data_with_new_type)


def train_data_type_ppl(fit_X_type=None, fit_Y_type=None, predict_X_type=None):
    data = [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [1.0, 2.0, 3.0], [2.0, 2.0, 2.0]]
    label = [1, 0, 1, 1]
    if fit_X_type == "sparse":
        model = Pipeline([Binner(), LightGbmClassifier(minimum_example_count_per_leaf=1)])
    else:
        model = Pipeline([Binner(), LogisticRegressionBinaryClassifier()])
    data_with_new_type = transform_data(data, fit_X_type)
    label_with_new_type = transform_data(label, fit_Y_type)
    model.fit(data_with_new_type, label_with_new_type)
    metrics, scores = model.test(
        data_with_new_type, label_with_new_type, output_scores=True)
    test_data_with_new_type = transform_data(data, predict_X_type)
    return model.predict(test_data_with_new_type), scores, metrics


class TestNumericDataType(unittest.TestCase):

    def test_check_datatype_single_list_list_array(self):
        result = train_data_type_single("list", "list", "array")
        assert len(result) == 4

    def test_check_datatype_single_list_list_dataframe(self):
        result = train_data_type_single("list", "list", "dataframe")
        assert len(result) == 4

    def test_check_datatype_single_list_list_list(self):
        result = train_data_type_single("list", "list", "list")
        assert len(result) == 4

    def test_check_datatype_single_array_list_array(self):
        result = train_data_type_single("array", "list", "array")
        assert len(result) == 4

    def test_check_datatype_single_array_array_dataframe(self):
        result = train_data_type_single("array", "array", "dataframe")
        assert len(result) == 4

    def test_check_datatype_single_array_list_list(self):
        result = train_data_type_single("array", "list", "list")
        assert len(result) == 4

    def test_check_datatype_single_dataframe_list_array(self):
        result = train_data_type_single("dataframe", "list", "array")
        assert len(result) == 4

    def test_check_datatype_single_dataframe_array_dataframe(self):
        result = train_data_type_single("dataframe", "array", "dataframe")
        assert len(result) == 4

    def test_check_datatype_single_dataframe_list_list(self):
        result = train_data_type_single("dataframe", "list", "list")
        assert len(result) == 4

    def test_check_datatype_single_sparse_list_sparse(self):
        result = train_data_type_single("sparse", "list", "sparse")
        assert len(result) == 4

    def test_check_datatype_ppl_list_list_array(self):
        result, scores, metrics = train_data_type_ppl("list", "list", "array")
        assert len(result) == 4
        assert scores is not None

    def test_check_datatype_ppl_list_list_dataframe(self):
        result, scores, metrics = train_data_type_ppl(
            "list", "list", "dataframe")
        assert len(result) == 4
        assert scores is not None

    def test_check_datatype_ppl_list_list_list(self):
        result, scores, metrics = train_data_type_ppl("list", "list", "list")
        assert len(result) == 4
        assert scores is not None

    def test_check_datatype_ppl_array_list_array(self):
        result, scores, metrics = train_data_type_ppl("array", "list", "array")
        assert len(result) == 4
        assert scores is not None

    def test_check_datatype_ppl_array_array_dataframe(self):
        result, scores, metrics = train_data_type_ppl(
            "array", "array", "dataframe")
        assert len(result) == 4
        assert scores is not None

    def test_check_datatype_ppl_array_list_list(self):
        result, scores, metrics = train_data_type_ppl("array", "list", "list")
        assert len(result) == 4
        assert scores is not None

    def test_check_datatype_ppl_dataframe_list_array(self):
        result, scores, metrics = train_data_type_ppl(
            "dataframe", "list", "array")
        assert len(result) == 4
        assert scores is not None

    def test_check_datatype_ppl_dataframe_array_dataframe(self):
        result, scores, metrics = train_data_type_ppl(
            "dataframe", "array", "dataframe")
        assert len(result) == 4
        assert scores is not None

    def test_check_datatype_ppl_dataframe_list_list(self):
        result, scores, metrics = train_data_type_ppl(
            "dataframe", "list", "list")
        assert len(result) == 4
        assert scores is not None

    def test_check_datatype_ppl_sparse_list_sparse(self):
        result, scores, metrics = train_data_type_ppl(
            "sparse", "list", "sparse")
        assert len(result) == 4
        assert scores is not None

    def test_check_series(self):
        data = pd.DataFrame(
            data=dict(a=[1.2, 2, 3], b=[2, 3, 4], c=[1, 2, 3], d=[2, 2, 2]))
        norm = MinMaxScaler() << "a"
        normalized1 = norm.fit_transform(data)
        normalized2 = norm.fit_transform(data['a'])
        assert_array_equal(normalized1['a'].values, normalized2['a'].values)


if __name__ == '__main__':
    unittest.main()
