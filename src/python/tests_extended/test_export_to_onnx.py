# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
"""
Verify onnx export and transform support
"""
import contextlib
import io
import json
import os
import sys
import six
import tempfile
import numpy as np
import pandas as pd
import pprint
import unittest

from nimbusml import Pipeline
from nimbusml.base_predictor import BasePredictor
from nimbusml.cluster import KMeansPlusPlus
from nimbusml.datasets import get_dataset
from nimbusml.datasets.image import get_RevolutionAnalyticslogo, get_Microsoftlogo
from nimbusml.decomposition import PcaTransformer, PcaAnomalyDetector
from nimbusml.ensemble import FastForestBinaryClassifier, FastTreesTweedieRegressor, LightGbmRanker
from nimbusml.feature_extraction.categorical import OneHotVectorizer, OneHotHashVectorizer
from nimbusml.feature_extraction.image import Loader, Resizer, PixelExtractor
from nimbusml.feature_extraction.text import NGramFeaturizer
from nimbusml.feature_extraction.text.extractor import Ngram
from nimbusml.feature_selection import CountSelector, MutualInformationSelector
from nimbusml.linear_model import (AveragedPerceptronBinaryClassifier,
                                   FastLinearBinaryClassifier,
                                   LinearSvmBinaryClassifier)
from nimbusml.multiclass import OneVsRestClassifier
from nimbusml.naive_bayes import NaiveBayesClassifier
from nimbusml.preprocessing import (TensorFlowScorer, FromKey, ToKey,
                                    DateTimeSplitter, OnnxRunner)
from nimbusml.preprocessing.filter import SkipFilter, TakeFilter, RangeFilter
from nimbusml.preprocessing.missing_values import Filter, Handler, Indicator
from nimbusml.preprocessing.normalization import Binner, GlobalContrastRowScaler, LpScaler
from nimbusml.preprocessing.schema import (ColumnConcatenator, TypeConverter,
                                           ColumnDuplicator, ColumnSelector, PrefixColumnConcatenator)
from nimbusml.preprocessing.text import CharTokenizer, WordTokenizer
from nimbusml.timeseries import (IidSpikeDetector, IidChangePointDetector,
                                 SsaSpikeDetector, SsaChangePointDetector,
                                 SsaForecaster)
from data_frame_tool import DataFrameTool as DFT

SHOW_ONNX_JSON = False
SHOW_TRANSFORMED_RESULTS = False
SHOW_FULL_PANDAS_OUTPUT = False

if SHOW_FULL_PANDAS_OUTPUT:
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 10000)

script_path = os.path.realpath(__file__)
script_dir = os.path.dirname(script_path)

#      Sepal_Length  Sepal_Width  Petal_Length  Petal_Width Label    Species  Setosa
# 0             5.1          3.5           1.4          0.2     0     setosa     1.0
# 1             4.9          3.0           1.4          0.2     0     setosa     1.0
iris_df = get_dataset("iris").as_df()
iris_df.drop(['Species'], axis=1, inplace=True)

iris_with_nan_df = iris_df.copy()
iris_with_nan_df.loc[1, 'Petal_Length'] = np.nan

iris_no_label_df = iris_df.drop(['Label'], axis=1)
iris_binary_df = iris_no_label_df.rename(columns={'Setosa': 'Label'})
iris_regression_df = iris_no_label_df.drop(['Setosa'], axis=1).rename(columns={'Petal_Width': 'Label'})

#   Unnamed: 0  education   age  parity  induced  case  spontaneous  stratum  pooled.stratum education_str
# 0           1        0.0  26.0     6.0      1.0   1.0          2.0      1.0     3.0        0-5yrs
# 1           2        0.0  42.0     1.0      1.0   1.0          0.0      2.0     1.0        0-5yrs
infert_df = get_dataset("infert").as_df()
infert_df.columns = [i.replace(': ', '') for i in infert_df.columns]
infert_df.rename(columns={'case': 'Label'}, inplace=True)

infert_onehot_df = (OneHotVectorizer() << 'education_str').fit_transform(infert_df)
infert_onehot_df['Label'] = infert_onehot_df['Label'].astype(np.uint32)

#     rank  group   carrier  price  Class  dep_day  nbr_stops  duration
# 0      2      1        AA    240      3        1          0      12.0
# 1      1      1        AA    300      3        0          1      15.0
file_path = get_dataset("gen_tickettrain").as_filepath()
gen_tt_df = pd.read_csv(file_path)
gen_tt_df['group'] = gen_tt_df['group'].astype(np.uint32)

#      Unnamed: 0  Label  Solar_R  Wind  Temp  Month  Day
# 0             1   41.0    190.0   7.4    67      5    1
# 1             2   36.0    118.0   8.0    72      5    2
airquality_df = get_dataset("airquality").as_df().fillna(0)
airquality_df = airquality_df[airquality_df.Ozone.notnull()]

#      Sentiment                                      SentimentText
# 0            1    ==RUDE== Dude, you are rude upload that carl...
# 1            1    == OK! ==  IM GOING TO VANDALIZE WILD ONES W...
file_path = get_dataset("wiki_detox_train").as_filepath()
wiki_detox_df = pd.read_csv(file_path, sep='\t')
wiki_detox_df = wiki_detox_df.head(10)

#                     Path  Label
# 0  C:\repo\src\python...   True
# 1  C:\repo\src\python...  False
image_paths_df = pd.DataFrame(data=dict(
    Path=[get_RevolutionAnalyticslogo(), get_Microsoftlogo()],
    Label=[True, False]))


SKIP = {
    'DatasetTransformer',
    'LightLda',
    'NGramExtractor', # Crashes
    'OneVsRestClassifier',
    'OnnxRunner',
    'Sentiment',
    'TensorFlowScorer',
    'TimeSeriesImputer',
    'TreeFeaturizer',
    'WordEmbedding',
    'Binner',
    'BootstrapSampler',
    'DateTimeSplitter',
    'EnsembleClassifier',
    'EnsembleRegressor',
    'FactorizationMachineBinaryClassifier',
    'Filter',
    'GamBinaryClassifier',
    'GamRegressor',
    'IidChangePointDetector',
    'IidSpikeDetector',
    'Loader',
    'LogMeanVarianceScaler',
    'OneHotHashVectorizer',
    'PcaAnomalyDetector',
    'PixelExtractor',
    'RangeFilter',
    'Resizer',
    'RobustScaler',
    'SkipFilter',
    'SsaChangePointDetector',
    'SsaForecaster',
    'SsaSpikeDetector',
    'SymSgdBinaryClassifier',
    'TakeFilter',
    'ToKeyImputer',
    'ToString',
    'EnsembleClassifier',
    'EnsembleRegressor',
    'CharTokenizer',
    'WordTokenizer',
    'MutualInformationSelector',
    'NaiveBayesClassifier',
    'CountSelector',
    'KMeansPlusPlus',
    'ToKey',
    'ColumnSelector',
    # below are fixed in ML.NET 1.5, broken 1.5.preview2
    'FastLinearClassifier',
    'LogisticRegressionClassifier',
    'OneHotVectorizer'
}

INSTANCES = {
    'AveragedPerceptronBinaryClassifier': AveragedPerceptronBinaryClassifier(
        feature=['education_str.0-5yrs', 'education_str.6-11yrs', 'education_str.12+ yrs']),
    'Binner': Binner(num_bins=3),
    'CharTokenizer': CharTokenizer(columns={'SentimentText_Transform': 'SentimentText'}),
    'ColumnConcatenator': ColumnConcatenator(columns={'Features': [
        'Sepal_Length',
        'Sepal_Width',
        'Petal_Length',
        'Petal_Width',
        'Setosa']}),
    'ColumnSelector': ColumnSelector(columns=['Sepal_Width', 'Sepal_Length']),
    'ColumnDuplicator': ColumnDuplicator(columns={'dup': 'Sepal_Width'}),
    'CountSelector': CountSelector(count=5, columns=['Sepal_Width']),
    'DateTimeSplitter': DateTimeSplitter(prefix='dt'),
    'FastForestBinaryClassifier': FastForestBinaryClassifier(feature=['Sepal_Width', 'Sepal_Length'],
                                                             label='Setosa'),
    'FastLinearBinaryClassifier': FastLinearBinaryClassifier(feature=['Sepal_Width', 'Sepal_Length'],
                                                             label='Setosa'),
    'FastTreesTweedieRegressor': FastTreesTweedieRegressor(label='Ozone'),
    'Filter': Filter(columns=[ 'Petal_Length', 'Petal_Width']),
    'FromKey': Pipeline([
        ToKey(columns=['Sepal_Length']),
        FromKey(columns=['Sepal_Length'])
    ]),
    # GlobalContrastRowScaler currently requires a vector input to work
    'GlobalContrastRowScaler': Pipeline([
        ColumnConcatenator() << {
            'concated_columns': [
                'Petal_Length',
                'Sepal_Width',
                'Sepal_Length']},
        GlobalContrastRowScaler(columns={'normed_columns': 'concated_columns'})
    ]),
    'Handler': Handler(replace_with='Mean', columns={'NewVals': 'Petal_Length'}),
    'IidSpikeDetector': IidSpikeDetector(columns=['Sepal_Length']),
    'IidChangePointDetector': IidChangePointDetector(columns=['Sepal_Length']),
    'Indicator': Indicator(columns={'Has_Nan': 'Petal_Length'}),
    'KMeansPlusPlus': KMeansPlusPlus(n_clusters=3, feature=['Sepal_Width', 'Sepal_Length']),
    'LightGbmRanker': LightGbmRanker(feature=['Class', 'dep_day', 'duration'],
                                     label='rank',
                                     group_id='group'),
    'Loader': Loader(columns={'ImgPath': 'Path'}),
    'LpScaler': Pipeline([
        ColumnConcatenator() << {
            'concated_columns': [
                'Petal_Length',
                'Sepal_Width',
                'Sepal_Length']},
        LpScaler(columns={'normed_columns': 'concated_columns'})
    ]),
    'MutualInformationSelector': Pipeline([
        ColumnConcatenator(columns={'Features': ['Sepal_Width', 'Sepal_Length', 'Petal_Width']}),
        MutualInformationSelector(
            columns='Features',
            label='Label',
            slots_in_output=2)  # only accept one column
    ]),
    'NaiveBayesClassifier': NaiveBayesClassifier(feature=['Sepal_Width', 'Sepal_Length']),
    'NGramFeaturizer': NGramFeaturizer(word_feature_extractor=Ngram(),
                                       char_feature_extractor=Ngram(), 
                                       keep_diacritics=True,
                                       columns={ 'features': ['SentimentText']}),
    'OneHotHashVectorizer': OneHotHashVectorizer(columns=['education_str']),
    'OneHotVectorizer': OneHotVectorizer(columns=['education_str']),
    'OneVsRestClassifier(AveragedPerceptronBinaryClassifier)': \
        OneVsRestClassifier(AveragedPerceptronBinaryClassifier(),
                            use_probabilities=True,
                            feature=['age',
                                     'education_str.0-5yrs',
                                     'education_str.6-11yrs',
                                     'education_str.12+ yrs'],
                            label='induced'),
    'OneVsRestClassifier(LinearSvmBinaryClassifier)': \
        OneVsRestClassifier(LinearSvmBinaryClassifier(),
                            use_probabilities=True,
                            feature=['age',
                                     'education_str.0-5yrs',
                                     'education_str.6-11yrs',
                                     'education_str.12+ yrs'],
                            label='induced'),
    'PcaAnomalyDetector': PcaAnomalyDetector(rank=3),
    'PcaTransformer':  PcaTransformer(rank=2),
    'PixelExtractor': Pipeline([
        Loader(columns={'ImgPath': 'Path'}),
        PixelExtractor(columns={'ImgPixels': 'ImgPath'}),
    ]),
    'PrefixColumnConcatenator': PrefixColumnConcatenator(columns={'Features': 'Sepal_'}),
    'Resizer': Pipeline([
        Loader(columns={'ImgPath': 'Path'}),
        Resizer(image_width=227, image_height=227,
                columns={'ImgResize': 'ImgPath'})
    ]),
    'SkipFilter': SkipFilter(count=5),
    'SsaSpikeDetector': SsaSpikeDetector(columns=['Sepal_Length'],
                                         seasonal_window_size=2),
    'SsaChangePointDetector': SsaChangePointDetector(columns=['Sepal_Length'],
                                                    seasonal_window_size=2),
    'SsaForecaster': SsaForecaster(columns=['Sepal_Length'],
                                   window_size=2,
                                   series_length=5,
                                   train_size=5,
                                   horizon=1),
    'RangeFilter': RangeFilter(min=5.0, max=5.1, columns=['Sepal_Length']),
    'TakeFilter': TakeFilter(count=100),
    'TensorFlowScorer': TensorFlowScorer(
        model_location=os.path.join(
            script_dir,
            '..',
            'nimbusml',
            'examples',
            'frozen_saved_model.pb'),
        columns={'c': ['a', 'b']}),
    'ToKey': ToKey(columns={'edu_1': 'education_str'}),
    'TypeConverter': TypeConverter(columns=['group'], result_type='R4'),
    'WordTokenizer': WordTokenizer(char_array_term_separators=[" "]) << {'wt': 'SentimentText'}
}

DATASETS = {
    'AveragedPerceptronBinaryClassifier': infert_onehot_df,
    'Binner': iris_no_label_df,
    'BootstrapSampler': infert_df,
    'CharTokenizer': wiki_detox_df,
    'EnsembleRegressor': iris_regression_df,
    'FactorizationMachineBinaryClassifier': iris_binary_df,
    'FastForestBinaryClassifier': iris_no_label_df,
    'FastForestRegressor': iris_regression_df,
    'FastLinearBinaryClassifier': iris_no_label_df,
    'FastLinearClassifier': iris_binary_df,
    'FastLinearRegressor': iris_regression_df,
    'FastTreesBinaryClassifier': iris_binary_df, 
    'FastTreesRegressor': iris_regression_df,
    'FastTreesTweedieRegressor': airquality_df,
    'Filter': iris_no_label_df,
    'GamBinaryClassifier': iris_binary_df,
    'GamRegressor': iris_regression_df,
    'GlobalContrastRowScaler': iris_df.astype(np.float32),
    'Handler': iris_with_nan_df,
    'Indicator': iris_with_nan_df,
    'LightGbmBinaryClassifier': iris_binary_df,
    'LightGbmRanker': gen_tt_df,
    'LinearSvmBinaryClassifier': iris_binary_df,
    'Loader': image_paths_df,
    'LogisticRegressionBinaryClassifier': iris_binary_df,
    'LogisticRegressionClassifier': iris_df,
    'LogMeanVarianceScaler': iris_no_label_df,
    'LpScaler': iris_no_label_df.drop(['Setosa'], axis=1).astype(np.float32),
    'MeanVarianceScaler': iris_no_label_df,
    'MinMaxScaler': iris_no_label_df,
    'NGramFeaturizer': wiki_detox_df,
    'OneHotHashVectorizer': infert_df,
    'OneHotVectorizer': infert_df,
    'OnlineGradientDescentRegressor': iris_regression_df,
    'OneVsRestClassifier(AveragedPerceptronBinaryClassifier)': infert_onehot_df,
    'OneVsRestClassifier(LinearSvmBinaryClassifier)': infert_onehot_df,
    'OrdinaryLeastSquaresRegressor': iris_regression_df,
    'PcaAnomalyDetector': iris_no_label_df,
    'PcaTransformer': iris_regression_df,
    'PixelExtractor': image_paths_df,
    'PoissonRegressionRegressor': iris_regression_df,
    'Resizer': image_paths_df,
    'SgdBinaryClassifier': iris_binary_df,
    'SymSgdBinaryClassifier': iris_binary_df,
    'ToKey': infert_df,
    'TypeConverter': gen_tt_df,
    'WordTokenizer': wiki_detox_df
}

EXPECTED_RESULTS = {
    'AveragedPerceptronBinaryClassifier': {'cols': [('PredictedLabel', 'PredictedLabel', 'PredictedLabel.output')]},
    'CharTokenizer': {'num_cols': 424, 'cols': 0},
    'ColumnConcatenator': {'num_cols': 11, 'cols': 0},
    'ColumnDuplicator': {'num_cols': 7, 'cols': 0},
    'ColumnSelector': {
        'num_cols': 2,
        'cols': [('Sepal_Width', 'Sepal_Width', 'Sepal_Width.output'),
                 ('Sepal_Length', 'Sepal_Length', 'Sepal_Length.output')]
    },
    #'EnsembleClassifier': {'cols': [('PredictedLabel', 'PredictedLabel')]},
    #'EnsembleRegressor': {'cols': [('Score', 'Score')]},
    'FastForestBinaryClassifier': {'cols': [('PredictedLabel', 'PredictedLabel', 'PredictedLabel.output')]},
    'FastForestRegressor': {'cols': [('Score', 'Score', 'Score.output')]},
    'FastLinearBinaryClassifier': {'cols': [('PredictedLabel', 'PredictedLabel', 'PredictedLabel.output')]},
    'FastLinearClassifier':  {'cols': [('PredictedLabel', 'PredictedLabel', 'PredictedLabel.output')]},
    'FastLinearRegressor': {'cols': [('Score', 'Score', 'Score.output')]},
    'FastTreesBinaryClassifier':  {'cols': [('PredictedLabel', 'PredictedLabel', 'PredictedLabel.output')]},
    'FastTreesRegressor': {'cols': [('Score', 'Score', 'Score.output')]},
    'FastTreesTweedieRegressor': {'cols': [('Score', 'Score', 'Score.output')]},
    'FromKey': {'num_cols': 6, 'cols': 0},
    'GlobalContrastRowScaler': {'num_cols': 12, 'cols': 0},
    'Handler': {'num_cols': 8, 'cols': 0},
    'Indicator': {'num_cols': 7, 'cols': 0},
    'KMeansPlusPlus':  {'cols': [('PredictedLabel', 'PredictedLabel', 'PredictedLabel.output')]},
    'LightGbmBinaryClassifier':  {'cols': [('PredictedLabel', 'PredictedLabel', 'PredictedLabel.output')]},
    'LightGbmClassifier':  {'cols': [('PredictedLabel', 'PredictedLabel', 'PredictedLabel.output')]},
    'LightGbmRanker': {'cols': [('Score', 'Score', 'Score.output')]},
    'LightGbmRegressor': {'cols': [('Score', 'Score', 'Score.output')]},
    'LinearSvmBinaryClassifier':  {'cols': [('PredictedLabel', 'PredictedLabel', 'PredictedLabel.output')]},
    'LogisticRegressionBinaryClassifier':  {'cols': [('PredictedLabel', 'PredictedLabel', 'PredictedLabel.output')]},
    'LogisticRegressionClassifier':  {'cols': [('PredictedLabel', 'PredictedLabel', 'PredictedLabel.output')]},
    'LpScaler': {'num_cols': 10, 'cols': 0},
    'MeanVarianceScaler': {'num_cols': 5, 'cols': 0},
    'MinMaxScaler': {'num_cols': 5, 'cols': 0},
    'MutualInformationSelector': {'num_cols': 8, 'cols': 0},
    'NGramFeaturizer': {'num_cols': 273, 'cols': 0},
    'NaiveBayesClassifier': {'cols': [('PredictedLabel', 'PredictedLabel', 'PredictedLabel.output')]},
    'OneHotVectorizer': {'num_cols': 12, 'cols': 0},
    'OneVsRestClassifier(AveragedPerceptronBinaryClassifier)': \
        {'cols': [('PredictedLabel', 'PredictedLabel', 'PredictedLabel.output')]},
    'OneVsRestClassifier(LinearSvmBinaryClassifier)': \
        {'cols': [('PredictedLabel', 'PredictedLabel', 'PredictedLabel.output')]},
    'OnlineGradientDescentRegressor': {'cols': [('Score', 'Score', 'Score.output')]},
    'OrdinaryLeastSquaresRegressor': {'cols': [('Score', 'Score', 'Score.output')]},
    'PcaTransformer': {'num_cols': 9, 'cols': 0},
    'PoissonRegressionRegressor': {'cols': [('Score', 'Score', 'Score.output')]},
    'PrefixColumnConcatenator': {'num_cols': 8, 'cols': 0},
    'SgdBinaryClassifier': {'cols': [('PredictedLabel', 'PredictedLabel', 'PredictedLabel.output')]},
    'SymSgdBinaryClassifier': {'cols': [('PredictedLabel', 'PredictedLabel', 'PredictedLabel.output')]},
    'ToKey': {'num_cols': 11, 'cols': 0},
    'TypeConverter': {'num_cols': 8, 'cols': 0},
    'WordTokenizer': {'num_cols': 73, 'cols': 0}
}

SUPPORTED_ESTIMATORS = {
    'AveragedPerceptronBinaryClassifier',
    'CharTokenizer',
    'ColumnConcatenator',
    'ColumnDuplicator',
    'ColumnSelector',
    'CountSelector',
    'EnsembleClassifier',
    'EnsembleRegressor',
    'FastForestBinaryClassifier',
    'FastForestRegressor',
    'FastLinearBinaryClassifier',
    'FastLinearClassifier',
    'FastLinearRegressor',
    'FastTreesBinaryClassifier',
    'FastTreesRegressor',
    'FastTreesTweedieRegressor',
    'FromKey',
    'GlobalContrastRowScaler',
    'Handler',
    'Indicator',
    'KMeansPlusPlus',
    'LightGbmBinaryClassifier',
    'LightGbmClassifier',
    'LightGbmRanker',
    'LightGbmRegressor',
    'LinearSvmBinaryClassifier',
    'LogisticRegressionBinaryClassifier',
    'LogisticRegressionClassifier',
    'LpScaler',
    'MeanVarianceScaler',
    'MinMaxScaler',
    'MutualInformationSelector',
    'NaiveBayesClassifier',
    'OneHotVectorizer',
    'OnlineGradientDescentRegressor',
    'OrdinaryLeastSquaresRegressor',
    'PcaTransformer',
    'PoissonRegressionRegressor',
    'SgdBinaryClassifier',
    'SymSgdBinaryClassifier',
    'ToKey',
    'TypeConverter',
    'WordTokenizer'
}


class CaptureOutputContext():
    """
    Context which can be used for
    capturing stdout and stderr. 
    """
    def __enter__(self):
        self.orig_stdout = sys.stdout
        self.orig_stderr = sys.stderr
        self.stdout_capturer = io.StringIO()
        self.stderr_capturer = io.StringIO()
        sys.stdout = self.stdout_capturer
        sys.stderr = self.stderr_capturer
        return self

    def __exit__(self, *args):
        sys.stdout = self.orig_stdout
        sys.stderr = self.orig_stderr
        self.stdout = self.stdout_capturer.getvalue()
        self.stderr = self.stderr_capturer.getvalue()

        if self.stdout:
            print(self.stdout)

        if self.stderr:
            print(self.stderr)

        # free up some memory
        del self.stdout_capturer
        del self.stderr_capturer


def get_tmp_file(suffix=None):
    fd, file_name = tempfile.mkstemp(suffix=suffix)
    fl = os.fdopen(fd, 'w')
    fl.close()
    return file_name


def get_file_size(file_path):
    file_size = 0
    try:
        file_size = os.path.getsize(file_path)
    except:
        pass
    return file_size


def load_json(file_path):
    with open(file_path) as f:
        lines = f.readlines()
        lines = [l for l in lines if not l.strip().startswith('#')]
        content_without_comments = '\n'.join(lines)
        return json.loads(content_without_comments)


def print_results(result_expected, result_onnx, result_onnx_ort):
    print("\nML.Net Output (Expected Result):")
    print(result_expected)
    if not isinstance(result_expected, pd.Series):
        print('Columns', result_expected.columns)

    print("\nOnnxRunner Result:")
    print(result_onnx)
    if not isinstance(result_onnx, pd.Series):
        print('Columns', result_onnx.columns)

    print("\nORT Result:")
    print(result_onnx_ort)
    if not isinstance(result_onnx_ort, pd.Series):
        print('Columns', result_onnx_ort.columns)

def validate_results(class_name, result_expected, result_onnx, result_ort):
    if not class_name in EXPECTED_RESULTS:
        raise RuntimeError("ERROR: ONNX model executed but no results specified for comparison.")

    if 'num_cols' in EXPECTED_RESULTS[class_name]:
        num_cols = EXPECTED_RESULTS[class_name]['num_cols']

        if len(result_expected.columns) != num_cols:
            raise RuntimeError("ERROR: The ML.Net output does not contain the expected number of columns.")

        if len(result_onnx.columns) != num_cols:
            raise RuntimeError("ERROR: The ONNX output does not contain the expected number of columns.")

        if len(result_ort.columns) != num_cols:
            raise RuntimeError("ERROR: The ORT output does not contain the expected number of columns.")

    col_tuples = EXPECTED_RESULTS[class_name]['cols']

    if isinstance(col_tuples, int):
        # If col_pairs is an int then slice the columns
        # based on the value and use those pairs for comparison
        col_tuples = list(zip(result_expected.columns[col_tuples:],
                              result_onnx.columns[col_tuples:],
                              result_ort.columns[col_tuples:]))

        if not col_tuples:
            raise RuntimeError("ERROR: no columns specified for comparison of results.")

    for col_tuple in col_tuples:
        try:
            col_expected = result_expected.loc[:, col_tuple[0]]
            col_onnx = result_onnx.loc[:, col_tuple[1]]
            col_ort = result_ort.loc[:, col_tuple[2]]

            if isinstance(col_expected.dtype, pd.api.types.CategoricalDtype):
                # ONNX does not export categorical columns so convert categorical
                # columns received from ML.Net back to the original values before
                # the comparison.
                col_expected = col_expected.astype(col_expected.dtype.categories.dtype)

            check_kwargs = {
                'check_names': False,
                'check_exact': False,
                'check_dtype': True,
                'check_less_precise': True
            }

            pd.testing.assert_series_equal(col_expected, col_onnx, **check_kwargs)
            pd.testing.assert_series_equal(col_expected, col_ort, **check_kwargs)

        except Exception as e:
            print(e)
            raise RuntimeError("ERROR: OnnxRunner result does not match expected result.")

    return True


def export_to_onnx(estimator, class_name):
    """
    Fit and test an estimator and determine
    if it supports exporting to the ONNX format.
    """
    onnx_path = get_tmp_file('.onnx')
    onnx_json_path = get_tmp_file('.onnx.json')

    output = None
    exported = False
    export_valid = False

    try:
        dataset = DATASETS.get(class_name, iris_df)
        estimator.fit(dataset)

        with CaptureOutputContext() as output:
            estimator.export_to_onnx(onnx_path,
                                     'com.microsoft.ml',
                                     dst_json=onnx_json_path,
                                     onnx_version='Stable')
    except Exception as e:
        print(e)

    onnx_file_size = get_file_size(onnx_path)
    onnx_json_file_size = get_file_size(onnx_json_path)

    if (output and
        (onnx_file_size != 0) and
        (onnx_json_file_size != 0) and
        (not 'cannot save itself as ONNX' in output.stdout) and
        (not 'Warning: We do not know how to save the predictor as ONNX' in output.stdout)):

        exported = True

        # print('ONNX model path:', onnx_path)

        if SHOW_ONNX_JSON:
            with open(onnx_json_path) as f:
                print(json.dumps(json.load(f), indent=4))

        # Verify that the output of the exported onnx graph
        # produces the same results as the standard estimators.
        if isinstance(estimator, BasePredictor):
            result_expected = estimator.predict(dataset)
        else:
            result_expected = estimator.transform(dataset)

        if isinstance(result_expected, pd.Series):
            result_expected = pd.DataFrame(result_expected)

        try:
            onnxrunner = OnnxRunner(model_file=onnx_path)
            result_onnx = onnxrunner.fit_transform(dataset)
            df_tool = DFT(onnx_path)
            result_ort = df_tool.execute(dataset, [])

            if SHOW_TRANSFORMED_RESULTS:
                print_results(result_expected, result_onnx, result_ort)

            export_valid = validate_results(class_name,
                                            result_expected,
                                            result_onnx,
                                            result_ort)
        except Exception as e:
            print(e)

    os.remove(onnx_path)
    os.remove(onnx_json_path)
    return {'exported': exported, 'export_valid': export_valid}


@unittest.skipIf(six.PY2, "Disabled as there is no onnxruntime package for Python 2.7")
class TestOnnxExport(unittest.TestCase):

    # This method is a static method of the class
    # because there were pytest fixture related
    # issues when the method was in the global scope.
    @staticmethod
    def generate_test_method(class_name, entry_point):
        def method(self):

            if class_name in INSTANCES:
                estimator = INSTANCES[class_name]
            else:
                mod = __import__('nimbusml.' + entry_point['Module'],
                                 fromlist=[str(class_name)])

                the_class = getattr(mod, class_name)
                estimator = the_class()

            result = export_to_onnx(estimator, class_name)
            assert result['exported']
            assert result['export_valid']

        return method

manifest_diff = os.path.join(script_dir, '..', 'tools', 'manifest_diff.json')
entry_points = load_json(manifest_diff)['EntryPoints']
entry_points.extend([
    {'NewName': 'OneVsRestClassifier(AveragedPerceptronBinaryClassifier)'},
    {'NewName': 'OneVsRestClassifier(LinearSvmBinaryClassifier)'}
])
entry_points = sorted(entry_points, key=lambda ep: ep['NewName'])

for entry_point in entry_points:
    class_name = entry_point['NewName']
    if class_name in SKIP:
        continue

    test_name = 'test_%s' % class_name.replace('(', '_').replace(')', '').lower()
    method = TestOnnxExport.generate_test_method(class_name, entry_point)
    setattr(TestOnnxExport, test_name, method)

if __name__ == '__main__':
    unittest.main()