###############################################################################
# EnsembleClassifier
import numpy as np
import pandas as pd
from nimbusml.datasets import get_dataset
from nimbusml.ensemble import EnsembleClassifier
from nimbusml.ensemble.feature_selector import RandomFeatureSelector
from nimbusml.ensemble.output_combiner import ClassifierVoting
from nimbusml.ensemble.subset_selector import RandomPartitionSelector
from nimbusml.ensemble.sub_model_selector import ClassifierBestDiverseSelector
from sklearn.model_selection import train_test_split

# use 'iris' data set to create test and train data
#    Sepal_Length  Sepal_Width  Petal_Length  Petal_Width Label Species  Setosa
# 0           5.1          3.5           1.4          0.2     0  setosa     1.0
# 1           4.9          3.0           1.4          0.2     0  setosa     1.0
np.random.seed(0)

df = get_dataset("iris").as_df()
df.drop(['Species'], inplace=True, axis=1)

X_train, X_test, y_train, y_test = \
    train_test_split(df.loc[:, df.columns != 'Label'], df['Label'])

# train a model with default sampling and ensembling parameters and score
ensemble_with_defaults = EnsembleClassifier(num_models=3).fit(X_train, y_train)

scores = ensemble_with_defaults.predict(X_test)
scores = pd.to_numeric(scores)

# evaluate the model
print('Accuracy:', np.mean(y_test == [i for i in scores]))
# Accuracy: 0.9473684210526315


# train a model with specific sampling and ensembling options and score
ensemble_with_options = EnsembleClassifier(
    num_models=3,
    sampling_type = RandomPartitionSelector(
        feature_selector=RandomFeatureSelector(
            features_selction_proportion=0.7)),
    sub_model_selector_type=ClassifierBestDiverseSelector(),
    output_combiner=ClassifierVoting()).fit(X_train, y_train)

scores = ensemble_with_options.predict(X_test)
scores = pd.to_numeric(scores)

# evaluate the model
# note that accuracy is lower than with defaults as this is a small dataset
# that we partition into 3 chunks for each classifier, which decreases model
# quality.
print('Accuracy:', np.mean(y_test == [i for i in scores]))
# Accuracy: 0.5789473684210527
