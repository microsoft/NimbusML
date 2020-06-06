# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
import unittest

import numpy
import pandas
from nimbusml import Pipeline, FileDataStream
from nimbusml.cluster import KMeansPlusPlus
from nimbusml.datasets import get_dataset
from nimbusml.decomposition import PcaTransformer
from nimbusml.ensemble import FastTreesBinaryClassifier, LightGbmRanker
from nimbusml.feature_extraction.categorical import OneHotVectorizer
from nimbusml.feature_extraction.text import NGramFeaturizer
from nimbusml.feature_extraction.text import WordEmbedding
from nimbusml.internal.utils.data_roles import Role
from nimbusml.internal.utils.data_schema import DataSchema
from nimbusml.linear_model import FastLinearRegressor
from nimbusml.preprocessing import ToKey
from nimbusml.preprocessing.missing_values import Filter
from nimbusml.preprocessing.normalization import MeanVarianceScaler
from nimbusml.preprocessing.schema import ColumnDropper as Drop, \
    ColumnConcatenator
from nimbusml.utils.exports import dot_export_pipeline
from sklearn.model_selection import train_test_split

_sentiments = """\
"ItemID"	"Sentiment"	"SentimentSource"	"SentimentText"	"RowNum"	\
"Positive"	"Train"	"Small"
1	0	"Sentiment140"	"is so sad for my APL friend............."	1	\
FALSE	TRUE	FALSE
2	0	"Sentiment140"	"I missed the New Moon trailer..."	2	FALSE    \
TRUE	FALSE
3	1	"Sentiment140"	"omg its already 7:30 :O"	3	TRUE	TRUE	\
FALSE
4	0	"Sentiment140"	".. Omgaga. Im sooo  im gunna CRy. I've been at\
 this dentist since 11.. I was suposed 2 just get a crown put on (\
 30mins)..."	4	FALSE	TRUE	FALSE
5	0	"Sentiment140"	"i think mi bf is cheating on me!!!       T_T"    \
5	FALSE	TRUE	FALSE
6	0	"Sentiment140"	"or i just worry too much?"	6	FALSE	TRUE	FALSE
7	1	"Sentiment140"	"Juuuuuuuuuuuuuuuuussssst Chillin!!"	7	\
TRUE	TRUE	FALSE
8	0	"Sentiment140"	"Sunny Again        Work Tomorrow  :-|       TV\
 Tonight"	8	FALSE	TRUE	FALSE
9	1	"Sentiment140"	"handed in my uniform today . i miss you\
 already"	9	TRUE	TRUE	FALSE
"""


class TestExports(unittest.TestCase):

    def test_object_parameters(self):
        obj1 = MeanVarianceScaler() << {'new_y': 'yy'}
        assert obj1._columns is not None
        obj2 = MeanVarianceScaler(columns={'new_y': 'yy'})
        assert obj1.get_params() == {
            'columns': {
                'new_y': 'yy'},
            'fix_zero': True,
            'max_training_examples': 1000000000,
            'use_cdf': False}
        assert obj1.get_params() == obj2.get_params()
        obj3 = FastLinearRegressor() << {
            'Feature': [
                'workclass',
                'education'],
            Role.Label: 'new_y'}
        exp = {'bias_learning_rate': 1.0,
               'caching': 'Auto',
               'convergence_check_frequency': None,
               'convergence_tolerance': 0.01,
               'feature': ['workclass', 'education'],
               'l1_threshold': None,
               'l2_regularization': None,
               'label': 'new_y',
               'loss': 'squared',
               'maximum_number_of_iterations': None,
               'normalize': 'Auto',
               'shuffle': True,
               'weight': None,
               'number_of_threads': None}
        assert obj3.get_params() == exp

    def test_object_clone(self):
        obj1 = MeanVarianceScaler() << {'new_y': 'yy'}
        obj2 = obj1.clone()
        pobj = obj1.get_params()
        assert pobj == obj2.get_params()
        assert '_columns' not in pobj
        assert 'columns' in pobj
        assert obj1._columns is not None
        assert pobj['columns'] is not None

    def test_pipeline_info(self):
        df = pandas.DataFrame(dict(education=['A', 'B', 'A', 'B', 'A'],
                                   workclass=['X', 'X', 'Y', 'Y', 'Y'],
                                   yy=[1.1, 2.2, 1.24, 3.4, 3.4]))

        exp = Pipeline([
            MeanVarianceScaler() << {'new_y': 'yy'},
            OneHotVectorizer() << ['workclass', 'education'],
            Drop() << 'yy',
            FastLinearRegressor() << {
                'Feature': ['workclass', 'education'], Role.Label: 'new_y'}
        ])

        infos = exp.get_fit_info(df)[0]
        for inf in infos:
            if 'operator' in inf:
                del inf['operator']
        exp = [{'name': None,
                'schema_after': ['education', 'workclass', 'yy'],
                'type': 'start',
                'outputs': ['education', 'workclass', 'yy']},
               {'name': 'TypeConverter', 'inputs': ['yy'],
                'outputs': ['new_y'],
                'schema_after': ['education', 'workclass', 'yy', 'new_y'],
                'type': 'transform'},
               {'name': 'MeanVarianceScaler', 'inputs': ['new_y'],
                'type': 'transform', 'outputs': ['new_y'],
                'schema_after': ['education', 'workclass', 'yy', 'new_y']},
               {'name': 'OneHotVectorizer',
                'inputs': ['workclass', 'education'], 'type': 'transform',
                'outputs': ['workclass', 'education'],
                'schema_after': ['education', 'workclass', 'yy', 'new_y']},
               {'name': 'ColumnDropper', 'type': 'transform',
                'schema_after': ['education', 'workclass', 'new_y'],
                'inputs': ['education', 'workclass', 'yy', 'new_y'],
                'outputs': ['education', 'workclass', 'new_y']},
               {'name': 'FastLinearRegressor',
                'inputs': ['Feature:education,workclass', 'Label:new_y'],
                'type': 'regressor', 'outputs': ['Score'],
                'schema_after': ['Score']}]
        if infos != exp:
            raise Exception(infos)

    def test_pipeline_info_strategy_previous(self):
        df = pandas.DataFrame(dict(education=['A', 'B', 'A', 'B', 'A'],
                                   workclass=['X', 'X', 'Y', 'Y', 'Y'],
                                   yy=[1.1, 2.2, 1.24, 3.4, 3.4]))
        X = df.drop('yy', axis=1)
        y = df['yy']

        exp = Pipeline([
            OneHotVectorizer() << ['workclass', 'education'],
            FastLinearRegressor()
        ])

        infos = exp.get_fit_info(X, y)[0]
        for inf in infos:
            if 'operator' in inf:
                del inf['operator']
        exp = [{'name': None,
                'schema_after': ['education', 'workclass', 'yy'],
                'type': 'start',
                'outputs': ['education', 'workclass', 'yy']},
               {'name': 'OneHotVectorizer',
                'inputs': ['workclass', 'education'], 'type': 'transform',
                'outputs': ['workclass', 'education'],
                'schema_after': ['education', 'workclass', 'yy']},
               {'name': 'FastLinearRegressor',
                'inputs': ['Feature:education,workclass', 'Label:yy'],
                'type': 'regressor', 'outputs': ['Score'],
                'schema_after': ['Score']}]
        assert infos == exp

    def test_pipeline_info_strategy_previous_drop(self):
        df = pandas.DataFrame(dict(education=['A', 'B', 'A', 'B', 'A'],
                                   workclass=['X', 'X', 'Y', 'Y', 'Y'],
                                   yy=[1.1, 2.2, 1.24, 3.4, 3.4]))
        X = df.drop('yy', axis=1)
        y = df['yy']

        exp = Pipeline([
            OneHotVectorizer() << ['workclass', 'education'],
            Drop() << ['education'],
            FastLinearRegressor()
        ])

        infos = exp.get_fit_info(X, y)[0]
        for inf in infos:
            if 'operator' in inf:
                del inf['operator']
        exp = [{'name': None,
                'schema_after': ['education', 'workclass', 'yy'],
                'type': 'start',
                'outputs': ['education', 'workclass', 'yy']},
               {'name': 'OneHotVectorizer',
                'inputs': ['workclass', 'education'], 'type': 'transform',
                'outputs': ['workclass', 'education'],
                'schema_after': ['education', 'workclass', 'yy']},
               {'name': 'ColumnDropper', 'type': 'transform',
                'schema_after': ['workclass', 'yy'],
                'inputs': ['education', 'workclass', 'yy'],
                'outputs': ['workclass', 'yy']},
               {'name': 'FastLinearRegressor',
                'inputs': ['Feature:workclass', 'Label:yy'],
                'type': 'regressor',
                'outputs': ['Score'], 'schema_after': ['Score']}]
        assert infos == exp

    def test_pipeline_info_strategy_previous_2(self):
        df = pandas.DataFrame(dict(education=['A', 'B', 'A', 'B', 'A'],
                                   workclass=['X', 'X', 'Y', 'Y', 'Y'],
                                   yy=[1.1, 2.2, 1.24, 3.4, 3.4]))
        X = df.drop('yy', axis=1)
        y = df['yy']

        exp = Pipeline([
            OneHotVectorizer() << ['workclass'],
            OneHotVectorizer() << ['education'],
            FastLinearRegressor()
        ])

        infos = exp.get_fit_info(X, y)[0]
        for inf in infos:
            if 'operator' in inf:
                del inf['operator']
        exp = [{'name': None,
                'schema_after': ['education',
                                 'workclass',
                                 'yy'],
                'type': 'start',
                'outputs': ['education',
                            'workclass',
                            'yy']},
               {'name': 'OneHotVectorizer',
                'inputs': ['workclass'],
                'type': 'transform',
                'outputs': ['workclass'],
                'schema_after': ['education',
                                 'workclass',
                                 'yy']},
               {'name': 'OneHotVectorizer',
                'inputs': ['education'],
                'type': 'transform',
                'outputs': ['education'],
                'schema_after': ['education',
                                 'workclass',
                                 'yy']},
               {'name': 'FastLinearRegressor',
                'inputs': ['Feature:education',
                           'Label:yy'],
                'type': 'regressor',
                'outputs': ['Score'],
                'schema_after': ['Score']}]
        assert infos == exp

    def test_pipeline_info_strategy_previous_2_accumulate(self):
        df = pandas.DataFrame(dict(education=['A', 'B', 'A', 'B', 'A'],
                                   workclass=['X', 'X', 'Y', 'Y', 'Y'],
                                   yy=[1.1, 2.2, 1.24, 3.4, 3.4]))
        X = df.drop('yy', axis=1)
        y = df['yy']

        exp = Pipeline([
            OneHotVectorizer() << ['workclass'],
            OneHotVectorizer() << ['education'],
            FastLinearRegressor()
        ])

        infos = exp.get_fit_info(X, y, iosklearn="accumulate")[0]
        for inf in infos:
            if 'operator' in inf:
                del inf['operator']
        exp = [{'name': None,
                'schema_after': ['education',
                                 'workclass',
                                 'yy'],
                'type': 'start',
                'outputs': ['education',
                            'workclass',
                            'yy']},
               {'name': 'OneHotVectorizer',
                'inputs': ['workclass'],
                'type': 'transform',
                'outputs': ['workclass'],
                'schema_after': ['education',
                                 'workclass',
                                 'yy']},
               {'name': 'OneHotVectorizer',
                'inputs': ['education'],
                'type': 'transform',
                'outputs': ['education'],
                'schema_after': ['education',
                                 'workclass',
                                 'yy']},
               {'name': 'FastLinearRegressor',
                'inputs': ['Feature:education,workclass',
                           'Label:yy'],
                'type': 'regressor',
                'outputs': ['Score'],
                'schema_after': ['Score']}]
        assert infos == exp

    def test_pipeline_exports(self):
        df = pandas.DataFrame(dict(education=['A', 'B', 'A', 'B', 'A'],
                                   workclass=['X', 'X', 'Y', 'Y', 'Y'],
                                   yy=[1.1, 2.2, 1.24, 3.4, 3.4]))

        exp = Pipeline([
            MeanVarianceScaler() << {'new_y': 'yy'},
            OneHotVectorizer() << ['workclass', 'education'],
            Drop() << 'yy',
            FastLinearRegressor() << {
                'Feature': ['workclass', 'education'], Role.Label: 'new_y'}
        ])

        for node in exp.nodes:
            if hasattr(node, 'label_column_name'):
                assert node.label_column_name == 'new_y'
        assert exp.nodes[-1].label_column_name == 'new_y'

        res = dot_export_pipeline(exp, df).strip("\n\r ")
        exp = """
                digraph{
                  orientation=portrait;
                  sch0[label="<f0> education|<f1> workclass|<f2> yy",
                  shape=record,fontsize=8];

                  node1[label="TypeConverter",shape=box,style="filled,
                  rounded",color=cyan,fontsize=12];
                  sch0:f2 -> node1;
                  sch1[label="<f0> new_y",shape=record,fontsize=8];
                  node1 -> sch1:f0;

                  node2[label="MeanVarianceScaler",shape=box,
                  style="filled,rounded",color=cyan,fontsize=12];
                  sch1:f0 -> node2;
                  sch2[label="<f0> new_y",shape=record,fontsize=8];
                  node2 -> sch2:f0;

                  node3[label="OneHotVectorizer",shape=box,
                  style="filled,rounded",color=cyan,fontsize=12];
                  sch0:f1 -> node3;
                  sch0:f0 -> node3;
                  sch3[label="<f0> workclass|<f1> education",
                  shape=record,fontsize=8];
                  node3 -> sch3:f0;
                  node3 -> sch3:f1;

                  node5[label="FastLinearRegressor",shape=box,
                  style="filled,rounded",color=yellow,fontsize=12];
                  sch3:f1 -> node5 [label="Feature",fontsize=8];
                  sch3:f0 -> node5 [label="Feature",fontsize=8];
                  sch2:f0 -> node5 [label="Label",fontsize=8];
                  sch5[label="<f0> Score",shape=record,fontsize=8];
                  node5 -> sch5:f0;
                }
                """.replace("                ", "").strip("\n\r ")
        if res.replace("\n", "").replace(" ", "") != exp.replace(
                "\n", "").replace(" ", ""):
            raise Exception(res)

    def test_pipeline_exports_complex(self):

        name = "test_pipeline_exports_complex.csv"
        with open(name, "w") as f:
            f.write(_sentiments)

        transform_1 = NGramFeaturizer() << {
            'transformed1': 'SentimentText'}
        transform_2 = OneHotVectorizer() << 'SentimentSource'
        transform_3 = ColumnConcatenator() << {
            'finalfeatures': [
                'transformed1',
                'SentimentSource']}
        algo = FastTreesBinaryClassifier() << {
            Role.Feature: 'finalfeatures', Role.Label: "Positive"}

        exp = Pipeline([transform_1, transform_2, transform_3, algo])

        stream = FileDataStream.read_csv(name, sep="\t")
        res = dot_export_pipeline(exp, stream).strip("\n\r ")
        exp = """
                digraph{
                  orientation=portrait;
                  sch0[label="<f0> ItemID|<f1> Sentiment|<f2> \
SentimentSource|<f3> SentimentText|<f4> RowNum|<f5> \
Positive|<f6> Train|<f7> Small",shape=record,fontsize=8];

                  node1[label="NGramFeaturizer",shape=box,style="filled,\
rounded",color=cyan,fontsize=12];
                  sch0:f3 -> node1;
                  sch1[label="<f0> transformed1|<f1> \
transformed1_TransformedText",shape=record,fontsize=8];
                  node1 -> sch1:f0;
                  node1 -> sch1:f1;

                  node2[label="OneHotVectorizer",shape=box,\
style="filled,rounded",color=cyan,fontsize=12];
                  sch0:f2 -> node2;
                  sch2[label="<f0> SentimentSource",shape=record,\
fontsize=8];
                  node2 -> sch2:f0;

                  node3[label="ColumnConcatenator",shape=box,\
style="filled,rounded",color=cyan,fontsize=12];
                  sch1:f0 -> node3;
                  sch2:f0 -> node3;
                  sch3[label="<f0> finalfeatures",shape=record,fontsize=8];
                  node3 -> sch3:f0;

                  node4[label="FastTreesBinaryClassifier",shape=box,\
style="filled,rounded",color=yellow,fontsize=12];
                  sch3:f0 -> node4 [label="Feature",fontsize=8];
                  sch0:f5 -> node4 [label="Label",fontsize=8];
                  sch4[label="<f0> PredictedLabel|<f1> \
PredictedProba|<f2> Score",shape=record,fontsize=8];
                  node4 -> sch4:f0;
                  node4 -> sch4:f1;
                  node4 -> sch4:f2;
                }
                """.replace("                ", "").strip("\n\r ")
        assert res == exp

    def test_get_fit_info_ranker(self):
        file_path = get_dataset("gen_tickettrain").as_filepath()
        file_schema = 'sep=, col=Label_1:R4:0 col=GroupId_2:TX:1 ' \
                      'col=Features_3:R4:3-5'
        train_stream = FileDataStream(file_path, schema=file_schema)
        pipeline = Pipeline(
            [
                ToKey() << {
                    'GroupId_2': 'GroupId_2'}, ColumnConcatenator() << {
                    'Features': ['Features_3']}, LightGbmRanker() << {
                    Role.Feature: 'Features', Role.Label: 'Label_1',
                    Role.GroupId: 'GroupId_2'}])

        info = pipeline.get_fit_info(train_stream)
        last = info[0][-1]
        inp = last['inputs']
        assert 'GroupId:GroupId_2' in inp

    def test_get_fit_info_clustering(self):
        X_train = pandas.DataFrame(data=dict(
            x=[0, 1, 2, 10, 11, 12, -10, -11, -12],
            y=[0, 1, 2, 10, 11, 12, -10, -11, -12],
            z=[0, 1, 2, 10, 11, 12, -10, -11, -12]))
        y_train = pandas.DataFrame(data=dict(
            clusterid=[0, 0, 0, 1, 1, 1, 2, 2, 2]))
        pipeline = Pipeline([KMeansPlusPlus(n_clusters=3)])
        pipeline.fit(X_train, y_train, verbose=0)
        scores = pipeline.predict(X_train)
        info = pipeline.get_fit_info(X_train, y_train)
        last = info[0][-1]
        out = last['outputs']
        assert out == ['PredictedLabel', 'Score.0', 'Score.1', 'Score.2']
        assert len(scores) == 9

    @unittest.skip('ML.NET does not have svm')
    def test_get_fit_info_anomaly(self):
        df = get_dataset("iris").as_df()
        df.drop(['Label', 'Setosa', 'Species'], axis=1, inplace=True)
        X_train, X_test = train_test_split(df)
        svm = Pipeline([OneClassSvmAnomalyDetector(  # noqa
            kernel=PolynomialKernel(a=1.0))])  # noqa
        svm.fit(X_train, verbose=0)
        scores = svm.predict(X_train)
        info = svm.get_fit_info(X_train)
        last = info[0][-1]
        out = last['outputs']
        assert len(scores) == len(X_train)
        assert out is not None

    def test_get_fit_info_fastl(self):
        train_file = get_dataset("airquality").as_filepath()
        schema = DataSchema.read_schema(train_file)
        data = FileDataStream(train_file, schema)

        pipeline = Pipeline([
            Filter(columns=['Ozone']),
            FastLinearRegressor(feature=['Solar_R', 'Temp'],
                                label='Ozone')])

        info = pipeline.get_fit_info(data)
        exp = [{'name': None,
                'outputs': ['Unnamed0',
                            'Ozone',
                            'Solar_R',
                            'Wind',
                            'Temp',
                            'Month',
                            'Day'],
                'schema_after': ['Unnamed0',
                                 'Ozone',
                                 'Solar_R',
                                 'Wind',
                                 'Temp',
                                 'Month',
                                 'Day'],
                'type': 'start'},
               {'inputs': ['Ozone'],
                'name': 'TypeConverter',
                'outputs': ['Ozone'],
                'schema_after': ['Unnamed0',
                                 'Ozone',
                                 'Solar_R',
                                 'Wind',
                                 'Temp',
                                 'Month',
                                 'Day'],
                'type': 'transform'},
               {'inputs': ['Ozone'],
                'name': 'Filter',
                'outputs': ['Ozone'],
                'schema_after': ['Unnamed0',
                                 'Ozone',
                                 'Solar_R',
                                 'Wind',
                                 'Temp',
                                 'Month',
                                 'Day'],
                'type': 'transform'}]
        for el in info[0]:
            if 'operator' in el:
                del el['operator']
        self.assertEqual(exp, info[0][:3])

    def test_word_embedding(self):

        ds_train = pandas.DataFrame(
            data=dict(
                description=[
                    "This is great",
                    "I hate it",
                    "Love it",
                    "Do not like it",
                    "Really like it",
                    "I hate it",
                    "I like it a lot",
                    "I kind of hate it",
                    "I do like it",
                    "I really hate it",
                    "It is very good",
                    "I hate it a bunch",
                    "I love it a bunch",
                    "I hate it",
                    "I like it very much",
                    "I hate it very much.",
                    "I really do love it",
                    "I really do hate it",
                    "Love it!",
                    "Hate it!",
                    "I love it",
                    "I hate it",
                    "I love it",
                    "I hate it",
                    "I love it"],
                like=[
                    True,
                    False,
                    True,
                    False,
                    True,
                    False,
                    True,
                    False,
                    True,
                    False,
                    True,
                    False,
                    True,
                    False,
                    True,
                    False,
                    True,
                    False,
                    True,
                    False,
                    True,
                    False,
                    True,
                    False,
                    True]))

        ng = NGramFeaturizer(columns=['description'], output_tokens_column_name='description_TransformedText')
        we = WordEmbedding(
            columns='description_TransformedText',
            model_kind='SentimentSpecificWordEmbedding')

        model = Pipeline([ng, we])
        dot_vis = dot_export_pipeline(model, ds_train)
        assert 'ch1[label="<f0> description|<f1> ' \
               'description_TransformedText"' in dot_vis

    def test_pipeline_pca(self):
        X = numpy.array([[1.0, 2, 3], [2, 3, 4], [3, 4, 5]])
        exp = Pipeline([PcaTransformer(rank=2)])
        infos = exp.get_fit_info(X)[0]
        for inf in infos:
            if 'operator' in inf:
                del inf['operator']
        exp = [{'name': None, 'schema_after': ['F0', 'F1', 'F2'],
                'type': 'start', 'outputs': ['F0', 'F1', 'F2']},
               {'name': 'TypeConverter', 'inputs': ['F0', 'F1', 'F2'],
                'type': 'transform',
                'outputs': ['F0', 'F1', 'F2'],
                'schema_after': ['F0', 'F1', 'F2']},
               {'name': 'PcaTransformer', 'inputs': ['temp_'],
                'type': 'transform', 'outputs': ['temp_'],
                'schema_after': ['F0', 'F1', 'F2', 'temp_']}]
        # This id depends on id(node), different at each execution.
        infos[-1]["inputs"] = ["temp_"]
        # This id depends on id(node), different at each execution.
        infos[-1]["outputs"] = ["temp_"]
        # This id depends on id(node), different at each execution.
        infos[-1]["schema_after"][-1] = ["temp_"]

        self.assertTrue(any(x != y for x, y in zip(exp, infos)))


if __name__ == "__main__":
    unittest.main()
