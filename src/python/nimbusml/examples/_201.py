from nimbusml import Pipeline, FileDataStream
from nimbusml.datasets import get_dataset
from nimbusml.feature_extraction.text import NGramFeaturizer
from nimbusml.feature_extraction.text.extractor import Ngram
from nimbusml.linear_model import AveragedPerceptronBinaryClassifier
from nimbusml.preprocessing.schema import ColumnConcatenator
import numpy as np
import pandas as pd

mlnet_model_file = "D:\\Data\\UCIAdult\\SampleBinaryClassification\\SampleBinaryClassification.Model\\MLNETModel.zip"
nimbus_fds_model_file = "D:\\Data\\UCIAdult\\SampleBinaryClassification\\SampleBinaryClassification.Model\\NimbusFDSModel.zip"
nimbus_df_model_file = "D:\\Data\\UCIAdult\\SampleBinaryClassification\\SampleBinaryClassification.Model\\NimbusDFModel.zip"
nimbus_scored_data_path = "D:\\Data\\UCIAdult\\scored_nimbus.tsv"

train_file = get_dataset('wiki_detox_train').as_filepath()
test_file = "D:\\Data\\UCIAdult\\test"

#train_df = pd.read_csv(train_file)
#train_df.columns = [i.replace(' ', '') for i in train_df.columns]
train_fds = FileDataStream.read_csv(train_file, sep='\t')

test_df = pd.read_csv(test_file)
test_df.columns = [i.replace(' ', '') for i in test_df.columns]
test_fds = FileDataStream.read_csv(test_file, sep='\t', numeric_dtype=np.float32)

pipe_fds = Pipeline([NGramFeaturizer(word_feature_extractor=Ngram(),
                                     columns={'features': ['SentimentText']}),
                     AveragedPerceptronBinaryClassifier(feature=['features'], label='Sentiment')])

pipe_fds.fit(train_fds)
##pipe_fds.save_model(nimbus_fds_model_file)
scores = pipe_fds.predict(train_fds)
print(scores.head())
#with pd.option_context('display.max_rows', None):
#    print(scores)

#print(sum(1*(scores['PredictedLabel'] == 1)))
#print(sum(1*(scores['PredictedLabel'] == 0)))


#scores = pipe_fds.predict(test_df)

#pipe_fds_loaded = Pipeline()
#pipe_fds_loaded.load_model(nimbus_fds_model_file)
##scores = pipe_fds_loaded.predict(test_fds)
#scores = pipe_fds_loaded.predict(test_df)

#pipe_df = pipe_fds.clone()
#pipe_df.fit(train_df)
##pipe_df.save_model(nimbus_df_model_file)
#scores = pipe_df.predict(test_fds)
#scores = pipe_df.predict(test_df)

#pipe_df_loaded = Pipeline()
#pipe_df_loaded.load_model(nimbus_df_model_file)
#scores = pipe_df_loaded.predict(test_fds)
#scores = pipe_df_loaded.predict(test_df)




#p = Pipeline()
#p.load_model(nimbus_fds_model_file)

#scores = p.predict(test_fds, verbose=100)
##scores['PredictedLabel'] = scores['PredictedLabel']*1
##scores.to_csv(nimbus_scored_data_path, sep="\t", index=False)
###print("scores from test fds")
#print(scores.head())
##with pd.option_context('display.max_rows', None, 'display.max_columns', None):
##    print(scores)

#probs = p.predict_proba(test_fds)
#print(probs.head())

#scores = p.predict(test_df, verbose=100)
#print("scores from test df")
#print(scores.head())
