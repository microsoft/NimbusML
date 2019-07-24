# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
"""
run check_estimator tests
"""
import json
import os

from nimbusml.ensemble import LightGbmBinaryClassifier
from nimbusml.ensemble import LightGbmClassifier
from nimbusml.ensemble import LightGbmRanker
from nimbusml.ensemble import LightGbmRegressor
from nimbusml.feature_extraction.text import NGramFeaturizer
from nimbusml.internal.entrypoints._ngramextractor_ngram import n_gram
from nimbusml.preprocessing import TensorFlowScorer
from nimbusml.preprocessing.filter import SkipFilter, TakeFilter
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
                           'check_regressors_int',
    # bug decision function shape should be 1
    # dimensional arrays, tolerance
    'FastLinearClassifier': 'check_classifiers_train',
    'FastForestRegressor': 'check_fit_score_takes_y',  # bug
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
    'LogisticRegressionClassifier': 'check_classifiers_train,',
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
}

OMITTED_CHECKS_TUPLE = (
    'OneHotHashVectorizer, FromKey, DssmFeaturizer, DnnFeaturizer, '
    'PixelExtractor, Loader, Resizer, \
                        GlobalContrastRowScaler, PcaTransformer, '
    'ColumnConcatenator, Sentiment, CharTokenizer, LightLda, '
    'NGramFeaturizer, \
                        WordEmbedding',
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
    'LightGbmBinaryClassifier': LightGbmBinaryClassifier(
        minimum_example_count_per_group=1, minimum_example_count_per_leaf=1),
    'LightGbmClassifier': LightGbmClassifier(
        minimum_example_count_per_group=1, minimum_example_count_per_leaf=1),
    'LightGbmRegressor': LightGbmRegressor(
        minimum_example_count_per_group=1, minimum_example_count_per_leaf=1),
    'LightGbmRanker': LightGbmRanker(
        minimum_example_count_per_group=1, minimum_example_count_per_leaf=1),
    'NGramFeaturizer': NGramFeaturizer(word_feature_extractor=n_gram()),
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


def my_import(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def load_json(file_path):
    with open(file_path) as f:
        lines = f.readlines()
        lines = [l for l in lines if not l.strip().startswith('#')]
        content_without_comments = '\n'.join(lines)
        return json.loads(content_without_comments)


skip_epoints = set(['OneVsRestClassifier', 'TreeFeaturizer'])
epoints = []
my_path = os.path.realpath(__file__)
my_dir = os.path.dirname(my_path)
manifest_diff_json = os.path.join(my_dir, '..', 'tools',
                                  'manifest_diff.json')
manifest_diff = load_json(manifest_diff_json)
for e in manifest_diff['EntryPoints']:
    if e['NewName'] not in skip_epoints:
        epoints.append((e['Module'], e['NewName']))

all_checks = {}
all_failed_checks = {}
all_passed_checks = {}
total_checks_passed = 0

print("total entrypoints: {}", len(epoints))

for e in epoints:
    checks = set()
    failed_checks = set()
    passed_checks = set()
    class_name = e[1]
    print("======== now Estimator is %s =========== " % class_name)
    # skip LighGbm for now, because of random crashes.
    if 'LightGbm' in class_name:
        continue
    # skip SymSgdBinaryClassifier for now, because of crashes.
    if 'SymSgdBinaryClassifier' in class_name:
        continue

    mod = __import__('nimbusml.' + e[0], fromlist=[str(class_name)])
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
        checks.add(check.__name__)
        try:
            check(class_name, estimator.clone())
            passed_checks.add(check.__name__)
            total_checks_passed = total_checks_passed + 1
        except Exception as e:
            failed_checks.add(check.__name__)

    if frozenset(checks) not in all_checks:
        all_checks[frozenset(checks)] = []
    all_checks[frozenset(checks)].append(class_name)

    if len(failed_checks) > 0:
        if frozenset(failed_checks) not in all_failed_checks:
            all_failed_checks[frozenset(failed_checks)] = []
        all_failed_checks[frozenset(failed_checks)].append(class_name)

    if frozenset(passed_checks) not in all_passed_checks:
        all_passed_checks[frozenset(passed_checks)] = []
    all_passed_checks[frozenset(passed_checks)].append(class_name)

if len(all_failed_checks) > 0:
    print("Following tests failed for components:")
    for key, value in all_failed_checks.items():
        print('========================')
        print(key)
        print(value)
    raise RuntimeError("estimator checks failed")
print("success, total checks passed %s ", total_checks_passed)
