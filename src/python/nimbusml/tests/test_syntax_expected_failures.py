# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
import unittest
import warnings

import pandas
import six
from nimbusml import Pipeline
from nimbusml.feature_extraction.categorical import OneHotVectorizer
from nimbusml.internal.utils.data_roles import Role
from nimbusml.linear_model import FastLinearRegressor
from nimbusml.pipeline import TrainedWarning

if six.PY2:
    pass
else:
    pass


class TestSyntaxExpectedFailures(unittest.TestCase):

    def test_syntax_onehot_trained(self):
        df = pandas.DataFrame(dict(edu=['A', 'B', 'A', 'B', 'A'],
                                   wk=['X', 'X', 'Y', 'Y', 'Y'],
                                   Label=[1.1, 2.2, 1.24, 3.4, 3.4]))

        onehot = (OneHotVectorizer() << 'edu').fit(df, verbose=0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pipe = Pipeline([onehot, FastLinearRegressor() << 'edu'])
            pipe.fit(df, verbose=0)
            assert len(w) == 1
            assert issubclass(w[-1].category, TrainedWarning)
            assert "already trained" in str(w[-1].message)
        pipe.fit(df, verbose=0)
        pipe = Pipeline([onehot.clone(), FastLinearRegressor() << 'edu'])

    def test_syntax_onehot_trained_all_rename(self):
        df = pandas.DataFrame(dict(edu=['A', 'B', 'A', 'B', 'A'],
                                   wk=['X', 'X', 'Y', 'Y', 'Y'],
                                   Label=[1.1, 2.2, 1.24, 3.4, 3.4]))

        onehot = (OneHotVectorizer() << {'edu2': 'edu'}).fit(df, verbose=0)
        df2 = onehot.transform(df)
        lr = (
            FastLinearRegressor() << ['edu2.A', 'edu2.B']).fit(
            df2,
            verbose=0)

        pipe = Pipeline([onehot.clone(), lr.clone() << ['edu2.A', 'edu2.B']])
        with self.assertRaises(RuntimeError):
            # 'Feature column 'edu2.A' not found
            pipe.fit(df, verbose=0)

        pipe = Pipeline([onehot.clone(), lr.clone() << ['edu2']])
        try:
            pipe.fit(df, verbose=0)
        except RuntimeError:
            # This should work!
            import pprint
            s = pprint.pformat(pipe.get_fit_info(df)[0])
            raise RuntimeError(s)

    def test_syntax_ambiguities_label(self):
        df = pandas.DataFrame(dict(edu=[1.12, 1.2, 2.24, 4.4, 5.4],
                                   wk=[1.1, 2.3, 1.56, 0.4, 2.4],
                                   Label=[1.1, 2.2, 1.24, 3.4, 3.4]))

        lr = FastLinearRegressor() << ['edu', 'wk']
        pipe = Pipeline([lr])
        assert pipe.nodes[0].has_defined_columns()
        pipe._check_ambiguities(df[['edu', 'wk']], df['Label'], None)
        pipe.fit(df, verbose=0)
        pipe.fit(df[['edu', 'wk']], df['Label'], verbose=0)

    def test_syntax_ambiguities_y(self):
        df = pandas.DataFrame(dict(edu=[1.12, 1.2, 2.24, 4.4, 5.4],
                                   wk=[1.1, 2.3, 1.56, 0.4, 2.4],
                                   y=[1.1, 2.2, 1.24, 3.4, 3.4]))

        lr = FastLinearRegressor()
        pipe = Pipeline([lr])

        assert not pipe.nodes[0].has_defined_columns()
        assert not pipe._is_fitted
        with self.assertRaises(RuntimeError):
            pipe.fit(df, verbose=0)
        assert not pipe._is_fitted
        assert pipe._run_time_error is not None
        pipe.fit(df[['edu', 'wk']], df['y'], verbose=0)
        assert pipe._is_fitted

    def test_syntax_ambiguities_yfit(self):
        df = pandas.DataFrame(dict(edu=[1.12, 1.2, 2.24, 4.4, 5.4],
                                   wk=[1.1, 2.3, 1.56, 0.4, 2.4],
                                   y=[1.1, 2.2, 1.24, 3.4, 3.4]))

        lr = FastLinearRegressor() << {
            Role.Feature: [
                'edu', 'wk'], Role.Label: 'y'}
        pipe = Pipeline([lr])
        assert pipe.nodes[0].has_defined_columns()
        with self.assertRaises(RuntimeError):
            pipe.fit(df[['edu', 'wk']], df['y'], verbose=0)


if __name__ == '__main__':
    unittest.main()
