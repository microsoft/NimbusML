import numpy as np
import pandas as pd
from nimbusml import Pipeline, FileDataStream
from nimbusml.datasets import get_dataset
from nimbusml.feature_extraction.categorical import OneHotVectorizer
from nimbusml.linear_model import LogisticRegressionBinaryClassifier
from nimbusml.preprocessing import DatasetTransformer
from nimbusml.preprocessing.schema import PrefixColumnConcatenator
from nimbusml.preprocessing.schema import ColumnDropper

path = get_dataset('infert').as_filepath()
data = FileDataStream.read_csv(path)
 
# train featurizer
featurization_pipeline = Pipeline([OneHotVectorizer(columns={'education': 'education'})])
featurization_pipeline.fit(data)
print(featurization_pipeline.get_schema())

# need to remove extra columns from csr matrix 
# and get csr_matrix featurized data
csr_featurization_pipeline = Pipeline([DatasetTransformer(featurization_pipeline.model), ColumnDropper() << ['case', 'row_num']])
sparse_featurized_data = csr_featurization_pipeline.fit_transform(data, as_csr=True)
# Note: the relative order of all columns is still the same.
print(csr_featurization_pipeline.get_schema())

concat_pipeline = Pipeline([DatasetTransformer(csr_featurization_pipeline.model), PrefixColumnConcatenator({'education': 'education.'})])
concat_pipeline.fit(data)
print(concat_pipeline.get_schema())

# train a featurizer + learner pipeline
# Note! order of feature columns on input to learner should be the same as in csr_matrix above
#feature_cols = ['parity', 'education', 'age', 'induced', 'spontaneous', 'stratum', 'pooled.stratum'] # 9 features
feature_cols = csr_featurization_pipeline.get_schema()
training_pipeline = Pipeline([DatasetTransformer(featurization_pipeline.model), LogisticRegressionBinaryClassifier(feature=feature_cols, label='case')])
training_pipeline.fit(data, output_predictor_model=True)
 
# load just a learner model
predictor_pipeline = Pipeline()
predictor_pipeline.load_model(training_pipeline.predictor_model)
print(predictor_pipeline.get_schema())

# use just a learner model on csr_matrix featurized data
predictor_pipeline.predict_proba(sparse_featurized_data.astype(np.float32))