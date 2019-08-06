###############################################################################
# EnsembleRegressor
import numpy as np
from nimbusml.datasets import get_dataset
from nimbusml.ensemble import EnsembleRegressor
from nimbusml.ensemble.feature_selector import RandomFeatureSelector
from nimbusml.ensemble.output_combiner import RegressorMedian
from nimbusml.ensemble.subset_selector import RandomPartitionSelector
from nimbusml.ensemble.sub_model_selector import RegressorBestDiverseSelector
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# use the built-in data set 'airquality' to create test and train data
#    Unnamed: 0  Ozone  Solar_R  Wind  Temp  Month  Day
# 0           1   41.0    190.0   7.4    67      5    1
# 1           2   36.0    118.0   8.0    72      5    2
np.random.seed(0)

df = get_dataset("airquality").as_df().fillna(0)
df = df[df.Ozone.notnull()]

X_train, X_test, y_train, y_test = train_test_split(
    df.loc[:, df.columns != 'Ozone'], df['Ozone'])

# train a model with default sampling and ensembling parameters and score
ensemble_with_defaults = EnsembleRegressor(num_models=3).fit(X_train, y_train)
scores = ensemble_with_defaults.predict(X_test)

# evaluate the model
print('R-squared fit:', r2_score(y_test, scores, ))
# R-squared fit: 0.12144964995862884


# train a model withe specific sampling and ensembling options and score
ensemble_with_options = EnsembleRegressor(
    num_models=3,
    sampling_type = RandomPartitionSelector(
        feature_selector=RandomFeatureSelector(
            features_selction_proportion=0.7)),
        sub_model_selector_type=RegressorBestDiverseSelector(),
        output_combiner=RegressorMedian()).fit(X_train, y_train)
scores = ensemble_with_options.predict(X_test)

# evaluate the model
# note that this is a worse fit than with defaults as this is a small dataset
# that we partition into 3 chunks for each regressor, which decreases model
# quality
print('R-squared fit:', r2_score(y_test, scores, ))
# R-squared fit: 0.027908675807698735