###############################################################################
# Pipeline with observation level feature contributions

# Scoring a dataset with a trained model produces a score, or prediction, for
# each example. To understand and explain these predictions it can be useful to
# inspect which features influenced them most significantly. This function
# computes a model-specific list of per-feature contributions to the score for
# each example. These contributions can be positive (they make the score
# higher) or negative (they make the score lower).

from nimbusml import Pipeline, FileDataStream
from nimbusml.datasets import get_dataset
from nimbusml.ensemble import FastTreesBinaryClassifier
from nimbusml.linear_model import LogisticRegressionBinaryClassifier

# data input (as a FileDataStream)
path = get_dataset('uciadult_train').as_filepath()

data = FileDataStream.read_csv(path)
print(data.head())
#    label  workclass     education  ... capital-loss hours-per-week
# 0      0    Private          11th  ...            0             40
# 1      0    Private       HS-grad  ...            0             50
# 2      1  Local-gov    Assoc-acdm  ...            0             40
# 3      1    Private  Some-college  ...            0             40
# 4      0          ?  Some-college  ...            0             30

# define the training pipeline with a linear model
lr_pipeline = Pipeline([LogisticRegressionBinaryClassifier(
    feature=['age', 'education-num', 'hours-per-week'], label='label')])

# train the model
lr_model = lr_pipeline.fit(data)

# For linear models, the contribution of a given feature is equal to the
# product of feature value times the corresponding weight. Similarly, for
# Generalized Additive Models (GAM), the contribution of a feature is equal to
# the shape function for the given feature evaluated at the feature value.
lr_feature_contributions = lr_model.get_feature_contributions(data)

# Print predictions with feature contributions, which give a relative measure
# of how much each feature impacted the Score.
print("========== Feature Contributions for Linear Model ==========")
print(lr_feature_contributions.head())
#   label  ... PredictedLabel     Score ... FeatureContributions.hours-per-week
# 0     0  ...              0 -2.010687 ...                            0.833069
# 1     0  ...              0 -1.216163 ...                            0.809928
# 2     1  ...              0 -1.248412 ...                            0.485957
# 3     1  ...              0 -1.132419 ...                            0.583148
# 4     0  ...              0 -1.969522 ...                            0.437361

# define the training pipeline with a tree model
tree_pipeline = Pipeline([FastTreesBinaryClassifier(
    feature=['age', 'education-num', 'hours-per-week'], label='label')])

# train the model
tree_model = tree_pipeline.fit(data)

# For tree-based models, the calculation of feature contribution essentially
# consists in determining which splits in the tree have the most impact on the
# final score and assigning the value of the impact to the features determining
# the split. More precisely, the contribution of a feature is equal to the
# change in score produced by exploring the opposite sub-tree every time a
# decision node for the given feature is encountered.
# 
# Consider a simple case with a single decision tree that has a decision node
# for the binary feature F1. Given an example that has feature F1 equal to
# true, we can calculate the score it would have obtained if we chose the
# subtree corresponding to the feature F1 being equal to false while keeping
# the other features constant. The contribution of feature F1 for the given
# example is the difference between the original score and the score obtained
# by taking the opposite decision at the node corresponding to feature F1. This
#  algorithm extends naturally to models with many decision trees.
tree_feature_contributions = tree_model.get_feature_contributions(data)

# Print predictions with feature contributions, which give a relative measure
# of how much each feature impacted the Score.
print("========== Feature Contributions for Tree Model ==========")
print(tree_feature_contributions.head())
#    label  ... PredictedLabel      Score ... FeatureContributions.hours-per-week
# 0      0  ...              0 -16.717360 ...                           -0.608664
# 1      0  ...              0  -7.688200 ...                           -0.541213
# 2      1  ...              1   1.571164 ...                            0.032862
# 3      1  ...              1   2.115638 ...                            0.537077
# 4      0  ...              0 -23.038410 ...                           -0.682764

