# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import unittest

try:
    # pandas 0.20.0+
    from pandas.api.types import is_string_dtype
except ImportError:
    def is_string_dtype(dt):
        return 'object' in str(dt) or "dtype('O')" in str(dt)

from nimbusml.tests.test_utils import get_accuracy, split_features_and_label, \
    check_supported_losses, check_unsupported_losses
from nimbusml.feature_extraction.categorical import OneHotVectorizer
from nimbusml.linear_model import AveragedPerceptronBinaryClassifier
from nimbusml.preprocessing.schema import ColumnDuplicator
from nimbusml.loss import Hinge, Poisson, SmoothedHinge, Squared, Tweedie
from nimbusml.datasets import get_dataset
from nimbusml import Pipeline
from sklearn.utils.testing import assert_greater


class TestAveragedPerceptronBinaryClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        df = get_dataset("infert").as_df()
        # remove : and ' ' from column names, and encode categorical column
        df.columns = [i.replace(': ', '') for i in df.columns]
        assert is_string_dtype(df['education_str'].dtype)
        df = (OneHotVectorizer() << ['education_str']).fit_transform(df)
        assert 'education_str' not in df.columns
        cls.X, cls.y = split_features_and_label(df, 'case')

    def test_averagedperceptron(self):
        accuracy = get_accuracy(self, AveragedPerceptronBinaryClassifier())
        # Accuracy depends on column Unnamed0 (index).
        assert_greater(accuracy, 0.98, "accuracy should be %s" % 0.98)

    def test_averagedperceptron_supported_losses(self):
        # bug: 'exp' fails on this test
        losses = [
            'log', 'smoothed_hinge', Hinge(
                margin=0.5), SmoothedHinge(
                smoothing_const=0.5)]
        check_supported_losses(
            self, AveragedPerceptronBinaryClassifier, losses, 0.95)

    def test_averagedperceptron_unsupported_losses(self):
        losses = [
            'squared',
            'poisson',
            'tweedie',
            Squared(),
            Poisson(),
            Tweedie(),
            100,
            'random_str']
        check_unsupported_losses(
            self, AveragedPerceptronBinaryClassifier, losses)

    def test_averagedperceptron_unsupported_losses_syntax(self):
        df = get_dataset("infert").as_df().drop('row_num', axis=1)
        X = df
        y = df['case']

        pipeline = Pipeline(
            [
                OneHotVectorizer(
                    columns={
                        'age1': 'age', 'parity1': 'parity',
                        'sp1': 'spontaneous'}),
                OneHotVectorizer(
                    columns={
                        'education_str': 'education_str'}),
                ColumnDuplicator(
                    columns={
                        'case2': 'case'}),
                AveragedPerceptronBinaryClassifier(
                    feature=[
                        'age1', 'education_str'], label='case')])

        try:
            model = pipeline.fit(X, y, verbose=0)
            raise AssertionError("same column name in X and y")
        except RuntimeError as e:
            assert "If any step in the pipeline has defined Label" in str(e)
        X = X.drop('case', axis=1)

        pipeline = Pipeline([
            OneHotVectorizer(
                columns={
                    'age1': 'age',
                    'parity1': 'parity',
                    'sp1': 'spontaneous'}),
            OneHotVectorizer(columns={'education_str': 'education_str'}),
            # ColumnDuplicator(columns={'case2': 'case'}), # does not work
            AveragedPerceptronBinaryClassifier(
                feature=['age1', 'education_str'], label='case')
        ])

        info = pipeline.get_fit_info(df)[0]
        assert info[-1]['inputs'] != ['Feature:Features', 'Label:case']

        model = pipeline.fit(df)
        y_pred_withpipeline = model.predict(X)
        assert set(y_pred_withpipeline.columns) == {
            'PredictedLabel', 'Probability', 'Score'}
        assert y_pred_withpipeline.shape == (248, 3)


if __name__ == '__main__':
    unittest.main()
