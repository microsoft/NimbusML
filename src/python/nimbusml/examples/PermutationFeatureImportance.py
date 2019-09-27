###############################################################################
# Permutation Feature Importance (PFI)

# Permutation feature importance (PFI) is a technique to determine the
# global importance of features in a trained machine learning model. The
# advantage of the PFI method is that it is model agnostic - it works
# with any model that can be evaluated - and it can use any dataset, not
# just the training set, to compute feature importance metrics.
        
# PFI works by taking a labeled dataset, choosing a feature, and
# permuting the values for that feature across all the examples, so that
# each example now has a random value for the feature and the original
# values for all other features. The evaluation metric (e.g. NDCG) is
# then calculated for this modified dataset, and the change in the
# evaluation metric from the original dataset is computed. The larger the
# change in the evaluation metric, the more important the feature is to
# the model. PFI works by performing this permutation analysis across all
# the features of a model, one after another.

# PFI is supported for binary classifiers, classifiers, regressors, and
# rankers.

from nimbusml import Pipeline, FileDataStream
from nimbusml.datasets import get_dataset
from nimbusml.ensemble import LightGbmRanker
from nimbusml.feature_extraction.categorical import OneHotVectorizer
from nimbusml.linear_model import LogisticRegressionBinaryClassifier, \
    FastLinearClassifier, FastLinearRegressor
from nimbusml.preprocessing import ToKey

# data input (as a FileDataStream)
adult_path = get_dataset('uciadult_train').as_filepath()
classification_data = FileDataStream.read_csv(adult_path)
print(classification_data.head())
#    label  workclass     education  ... capital-loss hours-per-week
# 0      0    Private          11th  ...            0             40
# 1      0    Private       HS-grad  ...            0             50
# 2      1  Local-gov    Assoc-acdm  ...            0             40
# 3      1    Private  Some-college  ...            0             40
# 4      0          ?  Some-college  ...            0             30

######################################
# PFI for Binary Classification models
######################################
# define the training pipeline with a binary classifier
binary_pipeline = Pipeline([
    OneHotVectorizer(columns=['education']),
    LogisticRegressionBinaryClassifier(
        feature=['age', 'education'], label='label')])

# train the model
binary_model = binary_pipeline.fit(classification_data)

print(binary_model.transform(classification_data).head())

# get permutation feature importance
binary_pfi = binary_model.permutation_feature_importance(classification_data)

# Print PFI for each feature, ordered by most important features w.r.t. AUC.
# Since AUC is an increasing metric, the highest negative changes indicate the
# most important features.
print("============== PFI for Binary Classification Model ==============")
print(binary_pfi.sort_values('AreaUnderRocCurve').head())
#               FeatureName  AreaUnderRocCurve  AreaUnderRocCurveStdErr  ...
# 0                     age          -0.081604                      0.0  ...
# 6   education.Prof-school          -0.012964                      0.0  ...
# 10    education.Doctorate          -0.012863                      0.0  ...
# 8     education.Bachelors          -0.010593                      0.0  ...
# 2       education.HS-grad          -0.005918


###############################
# PFI for Classification models
###############################
# define the training pipeline with a classifier
multiclass_pipeline = Pipeline([
    OneHotVectorizer(columns=['education']),
    FastLinearClassifier(feature=['age', 'education'], label='label')])

# train the model
multiclass_model = multiclass_pipeline.fit(classification_data)

# get permutation feature importance
multiclass_pfi = multiclass_model.permutation_feature_importance(classification_data)

# Print PFI for each feature, ordered by most important features w.r.t. Macro
# accuracy. Since Macro accuracy is an increasing metric, the highest negative
# changes indicate the most important features.
print("================== PFI for Classification Model ==================")
print(multiclass_pfi.sort_values('MacroAccuracy').head())
#                FeatureName  MacroAccuracy  MacroAccuracyStdErr  MicroAccuracy  ...
# 2        education.HS-grad      -0.024337                  0.0         -0.014  ...
# 9        education.Masters      -0.017843                  0.0         -0.004  ...
# 6    education.Prof-school      -0.016093                  0.0         -0.006  ...
# 10     education.Doctorate      -0.015189                  0.0         -0.014  ...
# 4   education.Some-college      -0.014794                  0.0         -0.004  ...

###########################
# PFI for Regression models
###########################
# load input data
infert_path = get_dataset('infert').as_filepath()
regression_data = FileDataStream.read_csv(infert_path)
print(regression_data.head())
#   age  case education  induced  parity  ... row_num  spontaneous  ...
# 0   26     1    0-5yrs        1       6 ...       1            2  ...
# 1   42     1    0-5yrs        1       1 ...       2            0  ...
# 2   39     1    0-5yrs        2       6 ...       3            0  ...
# 3   34     1    0-5yrs        2       4 ...       4            0  ...
# 4   35     1   6-11yrs        1       3 ...       5            1  ...

# define the training pipeline with a regressor
regression_pipeline = Pipeline([
    OneHotVectorizer(columns=['education']),
    FastLinearRegressor(feature=['induced', 'education'], label='age')
])

# train the model
regression_model = regression_pipeline.fit(regression_data)

# get permutation feature importance
regression_pfi = regression_model.permutation_feature_importance(regression_data)

# print PFI for each feaure, ordered by most important features w.r.t. MAE.
# Since MAE is an increasing metric, the highest positive changes indicate the
# most important features.
print("==================== PFI for Regression Model ====================")
print(regression_pfi.sort_values('MeanAbsoluteError', ascending=False).head())
#          FeatureName  MeanAbsoluteError  ...  RSquared  RSquaredStdErr
# 3  education.12+ yrs           0.206138  ... -0.065571             0.0
# 1   education.0-5yrs           0.128282  ... -0.053344             0.0
# 0            induced           0.101055  ... -0.032716             0.0
# 2  education.6-11yrs           0.002545  ... -0.006032             0.0

########################
# PFI for Ranking models
########################
# load input data
ticket_path = get_dataset('gen_tickettrain').as_filepath()
ranking_data = FileDataStream.read_csv(ticket_path)
print(ranking_data.head())
#    rank  group carrier  price  Class  dep_day  nbr_stops  duration
# 0     2      1      AA    240      3        1          0      12.0
# 1     1      1      AA    300      3        0          1      15.0
# 2     1      1      AA    360      3        0          2      18.0
# 3     0      1      AA    540      2        0          0      12.0
# 4     1      1      AA    600      2        0          1      15.0

# define the training pipeline with a ranker
ranking_pipeline = Pipeline([
    ToKey(columns=['group']),
    LightGbmRanker(feature=['Class', 'dep_day', 'duration'],
                   label='rank',
                   group_id='group')])

# train the model
ranking_model = ranking_pipeline.fit(ranking_data)

# get permutation feature importance
ranking_pfi = ranking_model.permutation_feature_importance(ranking_data)

# Print PFI for each feature, ordered by most important features w.r.t. DCG@1.
# Since DCG is an increasing metric, the highest negative changes indicate the
# most important features.
print("===================== PFI for Ranking Model =====================")
print(ranking_pfi.sort_values('DiscountedCumulativeGains.0').head())
#   FeatureName  DiscountedCumulativeGains.0  ...
# 0       Class                    -4.869096  ...
# 2    duration                    -2.344379  ...
# 1     dep_day                     0.000000  ...
