###############################################################################
# Pipeline with feature contributions
from nimbusml import Pipeline, FileDataStream
from nimbusml.datasets import get_dataset
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
# define the training pipeline
pipeline = Pipeline([LogisticRegressionBinaryClassifier(
    feature=['age', 'education-num', 'hours-per-week'], label='label')])

# train, predict, and evaluate
# TODO: Replace with CV
model = pipeline.fit(data)

feature_contributions = model.calculate_feature_contributions(
    data, output_scores=True)

# Print predictions with feature contributions, which give a relative measure
# of how much each feature impacted the Score.
print(feature_contributions.head())
#   label  ... PredictedLabel     Score ... FeatureContributions.hours-per-week
# 0        ...              0 -2.010687 ...                            0.833069
# 1        ...              0 -1.216163 ...                            0.809928
# 2        ...              0 -1.248412 ...                            0.485957
# 3        ...              0 -1.132419 ...                            0.583148
# 4        ...              0 -1.969522 ...                            0.437361
