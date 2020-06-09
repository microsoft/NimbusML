# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
"""
run check_estimator tests
"""
import json
import os
import distro
import unittest

from nimbusml.cluster import KMeansPlusPlus
from nimbusml.decomposition import FactorizationMachineBinaryClassifier
from nimbusml.ensemble import EnsembleClassifier
from nimbusml.ensemble import EnsembleRegressor
from nimbusml.ensemble import LightGbmBinaryClassifier
from nimbusml.ensemble import LightGbmClassifier
from nimbusml.ensemble import LightGbmRanker
from nimbusml.ensemble import LightGbmRegressor
from nimbusml.feature_extraction.text import NGramFeaturizer
from nimbusml.internal.entrypoints._ngramextractor_ngram import n_gram
from nimbusml.preprocessing import TensorFlowScorer, DateTimeSplitter
from nimbusml.linear_model import SgdBinaryClassifier
from nimbusml.preprocessing.filter import SkipFilter, TakeFilter
from nimbusml.preprocessing.normalization import RobustScaler
from nimbusml.timeseries import (IidSpikeDetector, IidChangePointDetector,
                                 SsaSpikeDetector, SsaChangePointDetector,
                                 SsaForecaster)
from sklearn.utils.estimator_checks import _yield_all_checks, MULTI_OUTPUT

this = os.path.abspath(os.path.dirname(__file__))
OMITTED_CHECKS = {
    # by design consistent input with model
    # don't accept randomly created inputs
    'TensorFlowScorer': 'check_dict_unchanged, '
                        'check_dont_overwrite_parameters, '
                        'check_dtype_object, '
                        'check_estimator_sparse_data, '
                        'check_estimators_dtypes, '
                        'check_estimators_fit_returns_self, '
                        'check_estimators_overwrite_params, '
                        'check_estimators_pickle, '
                        'check_fit1d_1feature, '
                        'check_fit2d_1feature, '
                        'check_fit2d_1sample, '
                        'check_fit2d_predict1d, '
                        'check_fit_score_takes_y, '
                        'check_pipeline_consistency, '
                        'check_transformer_data_not_an_array, '
                        'check_transformer_general',
    # by design non-determenistic output
    'BootstrapSampler': 'check_transformer_general, '
                        'check_transformer_data_not_an_array',
    # by design non-determenistic output
    'ColumnDropper': 'check_transformer_general, '
                     'check_transformer_data_not_an_array',
    # I8 should not have NA values
    'CountSelector':
        'check_estimators_dtypes',
    # DateTimeSplitter does not work with floating point types.
    'DateTimeSplitter':
        'check_transformer_general, check_pipeline_consistency'
        'check_estimators_pickle, check_estimators_dtypes'
        'check_dict_unchanged, check_dtype_object, check_fit_score_takes_y'
        'check_transformer_data_not_an_array, check_fit1d_1feature,'
        'check_fit2d_1feature, check_fit2d_predict1d, check_estimators_overwrite_params,'
        'check_estimator_sparse_data, check_fit2d_1sample, check_dont_overwrite_parameters,'
        'check_estimators_fit_returns_self',
    # by design returns smaller number of rows
    'SkipFilter': 'check_transformer_general, '
                  'check_transformer_data_not_an_array',
    # fix pending in PR, bug cant handle csr matrix
    'RangeFilter': 'check_estimators_dtypes, '
                   'check_estimator_sparse_data',
    # time series do not currently support sparse matrices
    'IidSpikeDetector': 'check_estimator_sparse_data',
    'IidChangePointDetector': 'check_estimator_sparse_data',
    'SsaSpikeDetector': 'check_estimator_sparse_data'
                        'check_fit2d_1sample', # SSA requires more than one sample
    'SsaChangePointDetector': 'check_estimator_sparse_data'
                              'check_fit2d_1sample', # SSA requires more than one sample
    'SsaForecaster': 'check_estimator_sparse_data'
                     'check_fit2d_1sample', # SSA requires more than one sample
    # bug, low tolerance
    'FastLinearRegressor': 'check_supervised_y_2d, '
                           'check_regressor_data_not_an_array, '
                           'check_regressors_int, '
                           # todo: investigate
                           'check_regressors_train',
    # bug decision function shape should be 1
    # dimensional arrays, tolerance
    'FastLinearClassifier': 'check_classifiers_train',
    'FastForestRegressor': 'check_fit_score_takes_y',  # bug
    'EnsembleClassifier': 'check_supervised_y_2d, '
                          'check_classifiers_train',
    'EnsembleRegressor': 'check_supervised_y_2d, '
                         'check_regressors_train',
    # bug in decision_function
    'FastTreesBinaryClassifier':
        'check_decision_proba_consistency',
    # I8 should not have NA values
    'Filter':
        'check_estimators_dtypes',
    # I8 should not have NA values
    'Handler':
        'check_estimators_dtypes',
    # I8 should not have NA values
    'Indicator':
        'check_estimators_dtypes',
    # tolerance
    'LogisticRegressionClassifier': 'check_classifiers_train',
    # todo: investigate
    'OnlineGradientDescentRegressor': 'check_regressors_train',
    # bug decision function shape, prediction bug
    'NaiveBayesClassifier':
        'check_classifiers_train, check_classifiers_classes',
    # bugs cant handle negative label
    'PoissonRegressionRegressor':
        'check_regressors_train, '
        'check_regressors_no_decision_function',
    'MutualInformationSelector':
        'check_dtype_object, check_estimators_dtypes, \
        check_estimators_pickle, '
        'check_transformer_data_not_an_array, '
        'check_transformer_general, \
        check_fit1d_1feature, check_fit_score_takes_y, '
        'check_fit2d_predict1d, '
        'check_dont_overwrite_parameters, \
        check_fit2d_1sample, check_dict_unchanged, '
        'check_estimators_overwrite_params, '
        'check_estimators_fit_returns_self, \
        check_fit2d_1feature, check_pipeline_consistency, '
        'check_estimator_sparse_data',
    # bug in decision_function
    'SymSgdBinaryClassifier':
        'check_decision_proba_consistency',
    # bug in decision_function
    'LightGbmClassifier': 'check_classifiers_train',
    # bug cant handle the data
    'LightGbmRegressor': 'check_fit2d_1sample',
    # bug, no attribute clusterer.labels_
    'KMeansPlusPlus': 'check_clustering',
    'LightGbmRanker':
        'check_classifiers_regression_target, '
        'check_pipeline_consistency, check_supervised_y_2d, '
        'check_classifiers_one_label, \
        check_classifiers_classes, check_fit2d_1feature, '
        'check_classifiers_train, check_fit2d_1sample, '
        'check_dont_overwrite_parameters,\
        check_classifier_data_not_an_array, check_dtype_object, '
        'check_fit_score_takes_y, check_estimators_dtypes,\
        check_estimators_nan_inf, check_dict_unchanged, '
        'check_fit1d_1feature, check_fit2d_predict1d, \
        check_estimators_overwrite_params, '
        'check_estimator_sparse_data, check_estimators_pickle, '
        'check_estimators_fit_returns_self',
    'PcaAnomalyDetector':
        'check_pipeline_consistency, check_supervised_y_2d, '
        'check_classifiers_classes, check_fit2d_1feature, \
        check_estimators_fit_returns_self, '
        'check_classifiers_train, '
        'check_dont_overwrite_parameters, \
        check_classifier_data_not_an_array, check_dtype_object,'
        ' check_fit_score_takes_y, check_estimators_dtypes,\
        check_dict_unchanged, check_fit1d_1feature, '
        'check_fit2d_predict1d, '
        'check_estimators_overwrite_params, \
        check_estimator_sparse_data, check_estimators_pickle, '
        'check_estimators_nan_inf',
    # RobustScaler does not support vectorized types
    'RobustScaler': 'check_estimator_sparse_data',
    'ToKeyImputer':
        'check_estimator_sparse_data, check_estimators_dtypes',
    # Most of these skipped tests are failing because the checks
    # require numerical types. ToString returns object types.
    # TypeError: ufunc 'isfinite' not supported for the input types
    'ToString': 'check_estimator_sparse_data, check_pipeline_consistency'
        'check_transformer_data_not_an_array, check_estimators_pickle'
        'check_transformer_general',
    'OrdinaryLeastSquaresRegressor': 'check_fit2d_1sample'
}

OMITTED_CHECKS_TUPLE = (
    'OneHotHashVectorizer, FromKey, DnnFeaturizer, '
    'PixelExtractor, Loader, Resizer, \
                        GlobalContrastRowScaler, PcaTransformer, '
    'ColumnConcatenator, Sentiment, CharTokenizer, LightLda, '
    'NGramFeaturizer, WordEmbedding, LpScaler, WordTokenizer'
    'NGramExtractor',
    'check_transformer_data_not_an_array, check_pipeline_consistency, '
    'check_fit2d_1feature, check_estimators_fit_returns_self,\
                       check_fit2d_1sample, '
    'check_dont_overwrite_parameters, '
    'check_dtype_object, check_fit_score_takes_y, '
    'check_estimators_dtypes, \
                       check_transformer_general, check_dict_unchanged, '
    'check_fit1d_1feature, check_fit2d_predict1d, '
    'check_estimators_overwrite_params, \
                       check_estimator_sparse_data, '
    'check_estimators_pickle')

OMITTED_CHECKS_ALWAYS = 'check_estimators_nan_inf'

NOBINARY_CHECKS = [
    'check_estimator_sparse_data',
    'check_dtype_object',
    'check_fit_score_takes_y',
    'check_fit2d_predict1d',
    'check_fit1d_1feature',
    'check_dont_overwrite_parameters',
    'check_supervised_y_2d',
    'check_estimators_fit_returns_self',
    'check_estimators_overwrite_params',
    'check_estimators_dtypes',
    'check_classifiers_classes',
    'check_classifiers_train']

INSTANCES = {
    'DateTimeSplitter': DateTimeSplitter(prefix='dt', columns=['F0']),
    'EnsembleClassifier': EnsembleClassifier(num_models=3),
    'EnsembleRegressor': EnsembleRegressor(num_models=3),
    'FactorizationMachineBinaryClassifier': FactorizationMachineBinaryClassifier(shuffle=False),
    'KMeansPlusPlus': KMeansPlusPlus(n_clusters=2),
    'LightGbmBinaryClassifier': LightGbmBinaryClassifier(
        minimum_example_count_per_group=1, minimum_example_count_per_leaf=1),
    'LightGbmClassifier': LightGbmClassifier(
        minimum_example_count_per_group=1, minimum_example_count_per_leaf=1),
    'LightGbmRegressor': LightGbmRegressor(
        minimum_example_count_per_group=1, minimum_example_count_per_leaf=1),
    'LightGbmRanker': LightGbmRanker(
        minimum_example_count_per_group=1, minimum_example_count_per_leaf=1),
    'NGramFeaturizer': NGramFeaturizer(word_feature_extractor=n_gram()),
    'RobustScaler': RobustScaler(scale=False),
    'SgdBinaryClassifier': SgdBinaryClassifier(number_of_threads=1, shuffle=False),
    'SkipFilter': SkipFilter(count=5),
    'TakeFilter': TakeFilter(count=100000),
    'IidSpikeDetector': IidSpikeDetector(columns=['F0']),
    'IidChangePointDetector': IidChangePointDetector(columns=['F0']),
    'SsaSpikeDetector': SsaSpikeDetector(columns=['F0'], seasonal_window_size=2),
    'SsaChangePointDetector': SsaChangePointDetector(columns=['F0'], seasonal_window_size=2),
    'SsaForecaster': SsaForecaster(columns=['F0'],
                                   window_size=2,
                                   series_length=5,
                                   train_size=5,
                                   horizon=1),
    'TensorFlowScorer': TensorFlowScorer(
        model_location=os.path.join(
            this,
            '..',
            'nimbusml',
            'examples',
            'frozen_saved_model.pb'),
        columns={'c': ['a', 'b']}),
}

MULTI_OUTPUT_EX = [
    'FastLinearClassifier',
    'FastLinearRegressor',
    'LogisticRegressionClassifier',
    'FastTreesRegressor',
    'FastForestRegressor',
    'FastTreesTweedieRegressor',
    'OneClassSvmAnomalyDetector',
    'NaiveBayesClassifier',
    'GamBinaryClassifier',
    'GamRegressor',
    'OnlineGradientDescentRegressor',
    'OrdinaryLeastSquaresRegressor',
    'PoissonRegressionRegressor',
    'SymSgdBinaryClassifier',
    'LightGbmClassifier',
    'LightGbmRegressor']

MULTI_OUTPUT.extend(MULTI_OUTPUT_EX)

skip_epoints = set([
    'OneVsRestClassifier',
    'TreeFeaturizer',
    # skip SymSgdBinaryClassifier for now, because of crashes.
    'SymSgdBinaryClassifier',
    'DatasetTransformer',
    'OnnxRunner',
    'TimeSeriesImputer'
])

if 'centos' in distro.linux_distribution(full_distribution_name=False)[0].lower():
    skip_epoints |= set([
        'DateTimeSplitter',
        'RobustScaler',
        'ToKeyImputer',
        'ToString'])


def load_json(file_path):
    with open(file_path) as f:
        lines = f.readlines()
        lines = [l for l in lines if not l.strip().startswith('#')]
        content_without_comments = '\n'.join(lines)
        return json.loads(content_without_comments)

def get_epoints():
    epoints = []
    my_path = os.path.realpath(__file__)
    my_dir = os.path.dirname(my_path)
    manifest_diff_json = os.path.join(my_dir, '..', 'tools',
                                      'manifest_diff.json')
    manifest_diff = load_json(manifest_diff_json)
    for e in manifest_diff['EntryPoints']:
        if (e['NewName'] not in skip_epoints) and ('LightGbm' not in e['NewName']):
            epoints.append((e['Module'], e['NewName']))

    return epoints


class TestEstimatorChecks(unittest.TestCase):
    # This method is a static method of the class
    # because there were pytest fixture related
    # issues when the method was in the global scope.
    @staticmethod
    def generate_test_method(epoint):
        def method(self):
            failed_checks = set()
            passed_checks = set()
            class_name = epoint[1]
            print("\n======== now Estimator is %s =========== " % class_name)

            mod = __import__('nimbusml.' + epoint[0], fromlist=[str(class_name)])
            the_class = getattr(mod, class_name)
            if class_name in INSTANCES:
                estimator = INSTANCES[class_name]
            else:
                estimator = the_class()

            if estimator._use_single_input_as_string():
                estimator = estimator << 'F0'

            for check in _yield_all_checks(class_name, estimator):
                # Skip check_dict_unchanged for estimators which
                # update the classes_ attribute. For more details
                # see https://github.com/microsoft/NimbusML/pull/200
                if (check.__name__ == 'check_dict_unchanged') and \
                    (hasattr(estimator, 'predict_proba') or
                     hasattr(estimator, 'decision_function')):
                    continue

                if check.__name__ in OMITTED_CHECKS_ALWAYS:
                    continue
                if 'Binary' in class_name and check.__name__ in NOBINARY_CHECKS:
                    continue
                if class_name in OMITTED_CHECKS and check.__name__ in \
                        OMITTED_CHECKS[class_name]:
                    continue
                if class_name in OMITTED_CHECKS_TUPLE[0] and check.__name__ in \
                        OMITTED_CHECKS_TUPLE[1]:
                    continue

                try:
                    check(class_name, estimator.clone())
                    passed_checks.add(check.__name__)
                except Exception as e:
                    failed_checks.add(check.__name__)

            if len(failed_checks) > 0:
                self.fail(msg=str(failed_checks))

        return method


for epoint in get_epoints():
    test_name = 'test_%s' % epoint[1].lower()
    method = TestEstimatorChecks.generate_test_method(epoint)
    setattr(TestEstimatorChecks, test_name, method)


if __name__ == '__main__':
    unittest.main()
