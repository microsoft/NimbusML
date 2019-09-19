import numpy as np
import pandas as pd
from nimbusml import Pipeline, FileDataStream
from nimbusml.datasets import get_dataset
from nimbusml.feature_extraction.categorical import OneHotVectorizer
from nimbusml.linear_model import LogisticRegressionBinaryClassifier, OnlineGradientDescentRegressor
from nimbusml.preprocessing.filter import RangeFilter
from nimbusml.preprocessing.schema import PrefixColumnConcatenator


seed = 0

train_data = {'c0': ['a', 'b', 'a', 'b'],
              'c1': [1, 2, 3, 4],
              'c2': [2, 3, 4, 5]}
train_df = pd.DataFrame(train_data).astype({'c1': np.float64,
                                            'c2': np.float64})

test_data = {'c0': ['a', 'b', 'b'],
             'c1': [1.5, 2.3, 3.7],
             'c2': [2.2, 4.9, 2.7]}
test_df = pd.DataFrame(test_data).astype({'c1': np.float64,
                                          'c2': np.float64})

# Create and fit a OneHotVectorizer transform using the
# training data and use it to transform the training data.
transform_pipeline = Pipeline([OneHotVectorizer() << 'c0'], random_state=seed)
transform_pipeline.fit(train_df)
df = transform_pipeline.transform(train_df)

# Create, fit and score with combined model. 
# Output predictor model separately.
combined_pipeline = Pipeline([OneHotVectorizer() << 'c0',
                    OnlineGradientDescentRegressor(label='c2')],
                    random_state=seed)
combined_pipeline.fit(train_df, output_predictor_model=True)
result_1 = combined_pipeline.predict(train_df)

# train ColumnConcatenatorV2 on featurized data
# specify single prefix instead of source columns
concat_pipeline = Pipeline([PrefixColumnConcatenator(columns={'c0': 'c0.'})])
concat_pipeline.fit(df)

# Load predictor pipeline
predictor_pipeline = Pipeline()
predictor_pipeline.load_model(combined_pipeline.predictor_model)

# combine concat and predictor models and score 
combined_predictor_pipeline = Pipeline.combine_models(concat_pipeline, 
                                            predictor_pipeline)
result_2 = combined_predictor_pipeline.predict(df)
                    