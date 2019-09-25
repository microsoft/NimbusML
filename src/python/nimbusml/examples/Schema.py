###############################################################################
# Pipeline
import numpy as np
import pandas as pd
from nimbusml import Pipeline, FileDataStream
from nimbusml.linear_model import FastLinearRegressor
from nimbusml.preprocessing.normalization import MeanVarianceScaler
from nimbusml.feature_extraction.text.extractor import Ngram
from nimbusml import FileDataStream, Pipeline
from nimbusml.feature_extraction.text import NGramFeaturizer

#df = pd.read_csv('E:\\tmp1\\columns.csv', comment='#', header=0, encoding='utf-8')

X_train = FileDataStream.read_csv("E:\\tmp1\\ag_news.csv")

pipe = Pipeline([
    NGramFeaturizer(columns={'Column2': 'Column2'}, 
                    word_feature_extractor = Ngram(ngram_length=3))
])

# max_num_terms = 10 000 000

pipe.fit(X_train)
print('fit done')

schema = pipe.get_schema()
print(schema)

