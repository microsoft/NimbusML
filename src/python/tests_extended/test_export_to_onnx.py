# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
"""
Verify onnx export support
"""
import contextlib
import io
import json
import os
import pandas
import sys
import tempfile
import numpy as np
import pandas as pd

from nimbusml import Pipeline
from nimbusml.cluster import KMeansPlusPlus
from nimbusml.datasets import get_dataset
from nimbusml.datasets.image import get_RevolutionAnalyticslogo, get_Microsoftlogo
from nimbusml.decomposition import PcaTransformer, PcaAnomalyDetector
from nimbusml.ensemble import FastForestBinaryClassifier, LightGbmRanker
from nimbusml.feature_extraction.categorical import OneHotVectorizer, OneHotHashVectorizer
from nimbusml.feature_extraction.image import Loader, Resizer, PixelExtractor
from nimbusml.feature_extraction.text import NGramFeaturizer
from nimbusml.feature_extraction.text.extractor import Ngram
from nimbusml.feature_selection import CountSelector, MutualInformationSelector
from nimbusml.linear_model import FastLinearBinaryClassifier
from nimbusml.naive_bayes import NaiveBayesClassifier
from nimbusml.preprocessing import TensorFlowScorer, FromKey, ToKey
from nimbusml.preprocessing.filter import SkipFilter, TakeFilter, RangeFilter
from nimbusml.preprocessing.missing_values import Handler, Indicator
from nimbusml.preprocessing.normalization import Binner, GlobalContrastRowScaler
from nimbusml.preprocessing.schema import (ColumnConcatenator, TypeConverter,
                                           ColumnDuplicator, ColumnSelector)
from nimbusml.preprocessing.text import CharTokenizer
from nimbusml.timeseries import (IidSpikeDetector, IidChangePointDetector,
                                 SsaSpikeDetector, SsaChangePointDetector,
                                 SsaForecaster)


SHOW_ONNX_JSON = False

script_path = os.path.realpath(__file__)
script_dir = os.path.dirname(script_path)

#      Sepal_Length  Sepal_Width  Petal_Length  Petal_Width Label    Species  Setosa
# 0             5.1          3.5           1.4          0.2     0     setosa     1.0
# 1             4.9          3.0           1.4          0.2     0     setosa     1.0
iris_df = get_dataset("iris").as_df()
iris_df.drop(['Species'], axis=1, inplace=True)

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

#     rank  group   carrier  price  Class  dep_day  nbr_stops  duration
# 0      2      1        AA    240      3        1          0      12.0
# 1      1      1        AA    300      3        0          1      15.0
file_path = get_dataset("gen_tickettrain").as_filepath()
gen_tt_df = pd.read_csv(file_path)
gen_tt_df['group'] = gen_tt_df['group'].astype(np.uint32)

#      Sentiment                                      SentimentText
# 0            1    ==RUDE== Dude, you are rude upload that carl...
# 1            1    == OK! ==  IM GOING TO VANDALIZE WILD ONES W...
file_path = get_dataset("wiki_detox_train").as_filepath()
wiki_detox_df = pd.read_csv(file_path, sep='\t')

#                     Path  Label
# 0  C:\repo\src\python...   True
# 1  C:\repo\src\python...  False
image_paths_df = pd.DataFrame(data=dict(
    Path=[get_RevolutionAnalyticslogo(), get_Microsoftlogo()],
    Label=[True, False]))


SKIP = {
    'DatasetTransformer',
    'LightLda',
    'OneVsRestClassifier',
    'Sentiment',
    'TensorFlowScorer',
    'TreeFeaturizer',
    'WordEmbedding'
}

INSTANCES = {
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
    'FastForestBinaryClassifier': FastForestBinaryClassifier(feature=['Sepal_Width', 'Sepal_Length'],
                                                             label='Setosa'),
    'FastLinearBinaryClassifier': FastLinearBinaryClassifier(feature=['Sepal_Width', 'Sepal_Length'],
                                                             label='Setosa'),
    'FromKey': Pipeline([
        ToKey(columns=['Setosa']),
        FromKey(columns=['Setosa'])
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
    'Handler': Handler(replace_with='Mean', columns={'NewVals': 'Sepal_Length'}),
    'IidSpikeDetector': IidSpikeDetector(columns=['Sepal_Length']),
    'IidChangePointDetector': IidChangePointDetector(columns=['Sepal_Length']),
    'Indicator': Indicator(columns={'Has_Nan': 'Petal_Length'}),
    'KMeansPlusPlus': KMeansPlusPlus(n_clusters=3, feature=['Sepal_Width', 'Sepal_Length']),
    'LightGbmRanker': LightGbmRanker(feature=['Class', 'dep_day', 'duration'],
                                     label='rank',
                                     group_id='group'),
    'Loader': Loader(columns={'ImgPath': 'Path'}),
    'MutualInformationSelector': Pipeline([
        ColumnConcatenator(columns={'Features': ['Sepal_Width', 'Sepal_Length', 'Petal_Width']}),
        MutualInformationSelector(
            columns='Features',
            label='Label',
            slots_in_output=2)  # only accept one column
    ]),
    'NaiveBayesClassifier': NaiveBayesClassifier(feature=['Sepal_Width', 'Sepal_Length']),
    'NGramFeaturizer': NGramFeaturizer(word_feature_extractor=Ngram(),
                                       columns={ 'features': ['SentimentText']}),
    'OneHotHashVectorizer': OneHotHashVectorizer(columns=['education_str']),
    'OneHotVectorizer': OneHotVectorizer(columns=['education_str']),
    'PcaAnomalyDetector': PcaAnomalyDetector(rank=3),
    'PcaTransformer': PcaTransformer(rank=3),
    'PixelExtractor': Pipeline([
        Loader(columns={'ImgPath': 'Path'}),
        PixelExtractor(columns={'ImgPixels': 'ImgPath'}),
    ]),
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
    'ToKey': ToKey(columns={'edu_1': 'education'}),
    'TypeConverter': TypeConverter(columns=['age'], result_type='R4')
}

DATASETS = {
    'AveragedPerceptronBinaryClassifier': infert_onehot_df,
    'Binner': iris_no_label_df,
    'BootstrapSampler': infert_df,
    'CharTokenizer': wiki_detox_df,
    'FactorizationMachineBinaryClassifier': iris_binary_df,
    'FastForestBinaryClassifier': iris_no_label_df,
    'FastForestRegressor': iris_regression_df,
    'FastLinearBinaryClassifier': iris_no_label_df,
    'FastLinearClassifier': iris_binary_df,
    'FastLinearRegressor': iris_regression_df,
    'FastTreesBinaryClassifier': iris_binary_df, 
    'FastTreesRegressor': iris_regression_df,
    'FastTreesTweedieRegressor': iris_regression_df,
    'GamBinaryClassifier': iris_binary_df,
    'GamRegressor': iris_regression_df,
    'GlobalContrastRowScaler': iris_df.astype(np.float32),
    'LightGbmRanker': gen_tt_df,
    'Loader': image_paths_df,
    'LogisticRegressionBinaryClassifier': iris_binary_df,
    'LogisticRegressionClassifier': iris_binary_df,
    'LogMeanVarianceScaler': iris_no_label_df,
    'MeanVarianceScaler': iris_no_label_df,
    'MinMaxScaler': iris_no_label_df,
    'NGramFeaturizer': wiki_detox_df,
    'OneHotHashVectorizer': infert_df,
    'OneHotVectorizer': infert_df,
    'OnlineGradientDescentRegressor': iris_regression_df,
    'OrdinaryLeastSquaresRegressor': iris_regression_df,
    'PcaAnomalyDetector': iris_no_label_df,
    'PcaTransformer': iris_no_label_df,
    'PixelExtractor': image_paths_df,
    'PoissonRegressionRegressor': iris_regression_df,
    'Resizer': image_paths_df,
    'SgdBinaryClassifier': iris_binary_df,
    'SymSgdBinaryClassifier': iris_binary_df,
    'ToKey': infert_df,
    'TypeConverter': infert_onehot_df
}

REQUIRES_EXPERIMENTAL = {
    'TypeConverter',
    'MeanVarianceScaler',
    'MinMaxScaler'
}

SUPPORTED_ESTIMATORS = {
    'ColumnConcatenator',
    'OneHotVectorizer',
    'MeanVarianceScaler',
    'MinMaxScaler',
    'TypeConverter'
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


def test_export_to_onnx(estimator, class_name):
    """
    Fit and test an estimator and determine
    if it supports exporting to the ONNX format.
    """
    onnx_path = get_tmp_file('.onnx')
    onnx_json_path = get_tmp_file('.onnx.json')

    output = None
    exported = False

    try:
        dataset = DATASETS.get(class_name, iris_df)
        estimator.fit(dataset)

        onnx_version = 'Experimental' if class_name in REQUIRES_EXPERIMENTAL else 'Stable'

        with CaptureOutputContext() as output:
            estimator.export_to_onnx(onnx_path,
                                     'com.microsoft.ml',
                                     dst_json=onnx_json_path,
                                     onnx_version=onnx_version)
    except Exception as e:
        print(e)

    onnx_file_size = get_file_size(onnx_path)
    onnx_json_file_size = get_file_size(onnx_json_path)

    if (output and
        (onnx_file_size != 0) and
        (onnx_json_file_size != 0) and
        (not 'cannot save itself as ONNX' in output.stdout)):
            exported = True

    if exported and SHOW_ONNX_JSON:
        with open(onnx_json_path) as f:
            print(json.dumps(json.load(f), indent=4))

    os.remove(onnx_path)
    os.remove(onnx_json_path)
    return exported


manifest_diff = os.path.join(script_dir, '..', 'tools', 'manifest_diff.json')
entry_points = load_json(manifest_diff)['EntryPoints']
entry_points = sorted(entry_points, key=lambda ep: ep['NewName'])

exportable_estimators = set()
exportable_experimental_estimators = set()
unexportable_estimators = set()

for entry_point in entry_points:
    class_name = entry_point['NewName']

    print('\n===========> %s' % class_name)

    if class_name in SKIP:
        print("skipped")
        continue

    mod = __import__('nimbusml.' + entry_point['Module'],
                     fromlist=[str(class_name)])

    if class_name in INSTANCES:
        estimator = INSTANCES[class_name]
    else:
        the_class = getattr(mod, class_name)
        estimator = the_class()

    result = test_export_to_onnx(estimator, class_name)

    if result:
        if class_name in REQUIRES_EXPERIMENTAL:
            exportable_experimental_estimators.add(class_name)
        else:
            exportable_estimators.add(class_name)

        print('Estimator successfully exported to ONNX.')

    else:
        unexportable_estimators.add(class_name)
        print('Estimator could NOT be exported to ONNX.')

print('\nThe following estimators were skipped: ', sorted(SKIP))
print('\nThe following estimators were successfully exported to ONNX: ', sorted(exportable_estimators))
print('\nThe following estimators were successfully exported to experimental ONNX: ', sorted(exportable_experimental_estimators))
print('\nThe following estimators could not be exported to ONNX: ', sorted(unexportable_estimators))

failed_estimators = SUPPORTED_ESTIMATORS.difference(exportable_estimators) \
                                        .difference(exportable_experimental_estimators)

if len(failed_estimators) > 0:
    print("The following tests failed exporting to onnx:", sorted(failed_estimators))
    raise RuntimeError("onnx export checks failed")

