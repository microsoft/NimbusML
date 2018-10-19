# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import unittest

from nimbusml import FileDataStream
from nimbusml import Pipeline
from nimbusml.datasets import get_dataset
from nimbusml.feature_extraction.categorical import OneHotHashVectorizer
from nimbusml.linear_model import FastLinearBinaryClassifier
from nimbusml.preprocessing.missing_values import Handler
from nimbusml.utils import get_X_y, evaluate_binary_classifier


class TestPipelineRoles(unittest.TestCase):

    def test_performance_syntax(self):
        train_file = get_dataset('uciadult_train').as_filepath()
        test_file = get_dataset('uciadult_test').as_filepath()
        file_schema = 'sep=, col=label:R4:0 col=Features:R4:9-14 ' \
                      'col=workclass:TX:1 col=education:TX:2 ' \
                      'col=marital-status:TX:3 col=occupation:TX:4 ' \
                      'col=relationship:TX:5 col=ethnicity:TX:6 ' \
                      'col=sex:TX:7 col=native-country-region:TX:8 header+'
        categorical_columns = [
            'workclass',
            'education',
            'marital-status',
            'occupation',
            'relationship',
            'ethnicity',
            'sex',
            'native-country-region']
        label_column = 'label'
        na_columns = ['Features']
        feature_columns_idv = na_columns + categorical_columns

        exp = Pipeline(
            [
                OneHotHashVectorizer(
                    columns=categorical_columns),
                Handler(
                    columns=na_columns),
                FastLinearBinaryClassifier(
                    feature=feature_columns_idv, label=label_column)])

        train_data = FileDataStream(train_file, schema=file_schema)
        exp.fit(train_data, label_column, verbose=0)
        print("train time %s" % exp._run_time)

        test_data = FileDataStream(test_file, schema=file_schema)
        out_data = exp.predict(test_data)
        print("predict time %s" % exp._run_time)

        (test, label_test) = get_X_y(test_file, label_column, sep=',')
        (acc1,
         auc1) = evaluate_binary_classifier(
            label_test.iloc[:, 0].values,
            out_data.loc[:, 'PredictedLabel'].values,
            out_data.loc[:, 'Probability'].values)

        print('ACC %s, AUC %s' % (acc1, auc1))

        exp = Pipeline([OneHotHashVectorizer() << categorical_columns,
                        Handler() << na_columns,
                        FastLinearBinaryClassifier() << feature_columns_idv])

        train_data = FileDataStream(train_file, schema=file_schema)
        exp.fit(train_data, label_column, verbose=0)
        print("train time %s" % exp._run_time)

        test_data = FileDataStream(test_file, schema=file_schema)
        out_data = exp.predict(test_data)
        print("predict time %s" % exp._run_time)

        (test, label_test) = get_X_y(test_file, label_column, sep=',')
        (acc2,
         auc2) = evaluate_binary_classifier(label_test.iloc[:,
                                            0].values,
                                            out_data.loc[:,
                                            'PredictedLabel'].values,
                                            out_data.loc[:,
                                            'Probability'].values)
        print('ACC %s, AUC %s' % (acc2, auc2))
        assert abs(acc1 - acc2) < 0.02
        assert abs(auc1 - auc2) < 0.02


if __name__ == '__main__':
    unittest.main()
