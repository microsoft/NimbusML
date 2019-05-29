# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
import os
import tempfile
import unittest

from nimbusml import FileDataStream
from nimbusml import Pipeline
from nimbusml.datasets import get_dataset
from nimbusml.ensemble import FastTreesBinaryClassifier
from nimbusml.feature_extraction.categorical import OneHotVectorizer
from nimbusml.linear_model import FastLinearBinaryClassifier
from nimbusml.utils import check_accuracy, get_X_y
from sklearn.utils.testing import assert_raises_regex, assert_equal, assert_true

train_file = get_dataset("uciadult_train").as_filepath()
test_file = get_dataset("uciadult_test").as_filepath()
categorical_columns = [
    'workclass',
    'education',
    'marital-status',
    'occupation',
    'relationship',
    'ethnicity',
    'sex',
    'native-country-region']
file_schema = 'sep=, col=label:R4:0 col=Features:R4:9-14 col=workclass:TX:1 ' \
              'col=education:TX:2 col=marital-status:TX:3 ' \
              'col=occupation:TX:4 col=relationship:TX:5 col=ethnicity:TX:6 ' \
              'col=sex:TX:7 col=native-country-region:TX:8 header+'
label_column = 'label'


class TestUciAdult(unittest.TestCase):

    def test_file_no_schema(self):
        pipeline = Pipeline([OneHotVectorizer() << categorical_columns,
                             FastLinearBinaryClassifier(number_of_threads=1,
                                                        shuffle=False)])
        assert_raises_regex(
            TypeError,
            'Filenames are not allowed',
            pipeline.fit,
            train_file,
            y=label_column)
        assert_raises_regex(
            ValueError,
            'Model is not fitted',
            pipeline.predict,
            test_file,
            y=label_column)

    def test_linear_file(self):
        pipeline = Pipeline([OneHotVectorizer() << categorical_columns,
                             FastLinearBinaryClassifier(number_of_threads=1,
                                                        shuffle=False)])

        train_stream = FileDataStream(train_file, schema=file_schema)
        assert 'sep' in train_stream.schema.options
        assert 'header' in train_stream.schema.options
        pipeline.fit(train_stream, label_column)
        test_stream = FileDataStream(test_file, schema=file_schema)
        out_data = pipeline.predict(test_stream)
        check_accuracy(test_file, label_column, out_data, 0.65)

    def test_linear_file_role(self):
        pipeline = Pipeline([OneHotVectorizer() << categorical_columns,
                             FastLinearBinaryClassifier(number_of_threads=1,
                                                        shuffle=False)])
        train_stream = FileDataStream(train_file, schema=file_schema)
        train_stream._set_role('Label', label_column)
        pipeline.fit(train_stream)
        test_stream = FileDataStream(test_file, schema=file_schema)
        out_data = pipeline.predict(test_stream)
        check_accuracy(test_file, label_column, out_data, 0.65)

    def test_linear_file_role2(self):
        pipeline = Pipeline([OneHotVectorizer() << categorical_columns,
                             FastLinearBinaryClassifier(
                                 number_of_threads=1, shuffle=False) << {
                                 'Label': label_column}])
        train_stream = FileDataStream(train_file, schema=file_schema)
        train_stream._set_role('Label', label_column)
        pipeline.fit(train_stream)
        test_stream = FileDataStream(test_file, schema=file_schema)
        out_data = pipeline.predict(test_stream)
        check_accuracy(test_file, label_column, out_data, 0.65)

    def test_trees_file(self):
        pipeline = Pipeline([OneHotVectorizer() << categorical_columns,
                             FastTreesBinaryClassifier() << {
                                 'Label': label_column}])
        train_stream = FileDataStream(train_file, schema=file_schema)
        pipeline.fit(train_stream, label_column)
        test_stream = FileDataStream(test_file, schema=file_schema)
        out_data = pipeline.predict(test_stream)
        check_accuracy(test_file, label_column, out_data, 0.65)

    def test_linear(self):
        (train, label) = get_X_y(train_file, label_column, sep=',')
        (test, label1) = get_X_y(test_file, label_column, sep=',')
        pipeline = Pipeline([OneHotVectorizer() << categorical_columns,
                             FastLinearBinaryClassifier(number_of_threads=1,
                                                        shuffle=False)])
        pipeline.fit(train, label)
        out_data = pipeline.predict(test)
        check_accuracy(test_file, label_column, out_data, 0.65)

    def test_linear_with_train_schema(self):
        (train, label) = get_X_y(train_file, label_column, sep=',')
        (test, label1) = get_X_y(test_file, label_column, sep=',')
        pipeline = Pipeline([OneHotVectorizer() << categorical_columns,
                             FastLinearBinaryClassifier(number_of_threads=1,
                                                        shuffle=False)])
        pipeline.fit(train, label)
        out_data = pipeline.predict(test)
        check_accuracy(test_file, label_column, out_data, 0.65)

    def test_linear_with_test_schema(self):
        (train, label) = get_X_y(train_file, label_column, sep=',')
        (test, label1) = get_X_y(test_file, label_column, sep=',')
        pipeline = Pipeline([OneHotVectorizer() << categorical_columns,
                             FastLinearBinaryClassifier(number_of_threads=1,
                                                        shuffle=False)])
        pipeline.fit(train, label)
        out_data = pipeline.predict(test)
        check_accuracy(test_file, label_column, out_data, 0.65)

    def test_linear_with_train_test_schema(self):
        (train, label) = get_X_y(train_file, label_column, sep=',')
        (test, label1) = get_X_y(test_file, label_column, sep=',')
        pipeline = Pipeline([OneHotVectorizer() << categorical_columns,
                             FastLinearBinaryClassifier(number_of_threads=1,
                                                        shuffle=False)])
        pipeline.fit(train, label)
        out_data = pipeline.predict(test)
        check_accuracy(test_file, label_column, out_data, 0.65)

    def test_trees(self):
        (train, label) = get_X_y(train_file, label_column, sep=',')
        (test, label1) = get_X_y(test_file, label_column, sep=',')
        pipeline = Pipeline([OneHotVectorizer() << categorical_columns,
                             FastTreesBinaryClassifier()])
        pipeline.fit(train, label)
        out_data = pipeline.predict(test)
        check_accuracy(test_file, label_column, out_data, 0.65)

    def test_experiment_loadsavemodel(self):
        (train, label) = get_X_y(train_file, label_column, sep=',')
        (test, label1) = get_X_y(test_file, label_column, sep=',')
        cat = OneHotVectorizer() << categorical_columns
        ftree = FastTreesBinaryClassifier()
        pipeline = Pipeline([cat, ftree])
        pipeline.fit(train, label)
        metrics1, scores1 = pipeline.test(
            test, label1, 'binary', output_scores=True)
        sum1 = metrics1.sum().sum()
        (fd, modelfilename) = tempfile.mkstemp(suffix='.model.bin')
        fl = os.fdopen(fd, 'w')
        fl.close()
        pipeline.save_model(modelfilename)

        pipeline2 = Pipeline()
        pipeline2.load_model(modelfilename)
        metrics2, scores2 = pipeline2.test(
            test, label1, 'binary', output_scores=True)
        sum2 = metrics2.sum().sum()

        assert_equal(
            sum1,
            sum2,
            "model metrics don't match after loading model")

    def test_parallel(self):
        (train, label) = get_X_y(train_file, label_column, sep=',')
        cat = OneHotVectorizer() << categorical_columns
        ftree = FastTreesBinaryClassifier()
        pipeline = Pipeline([cat, ftree])

        result = pipeline.fit(train, label, parallel=8)
        result2 = pipeline.fit(train, label, parallel=1)
        assert_true(result == result2)

if __name__ == '__main__':
    unittest.main()
