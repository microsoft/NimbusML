###############################################################################
# CV - cross-validate data
import numpy as np
from nimbusml import Pipeline, FileDataStream, DataSchema
from nimbusml.datasets import get_dataset
from nimbusml.feature_extraction.categorical import OneHotVectorizer
from nimbusml.linear_model import LogisticRegressionClassifier, \
    FastLinearRegressor
from nimbusml.model_selection import CV
from nimbusml.preprocessing.missing_values import Indicator, Handler

# Case 1: Default usage of CV

path = get_dataset('infert').as_filepath()
schema = DataSchema.read_schema(path, numeric_dtype=np.float32)
data = FileDataStream.read_csv(path, schema=schema)

pipeline = Pipeline([
    OneHotVectorizer(columns={'edu': 'education'}),
    LogisticRegressionClassifier(feature=['age', 'spontaneous', 'edu'],
                                 label='induced')])

# Do 3-fold cross-validation
cv_results = CV(pipeline).fit(data, cv=3)

# print summary statistic of metrics
print(cv_results['metrics_summary'])

# print metrics for all folds
print(cv_results['metrics'])

# print confusion matrix for fold 1
cm = cv_results['confusion_matrix']
print(cm[cm.Fold == 1])

# Case 2: Using CV with split_start option

path = get_dataset("airquality").as_filepath()
schema = DataSchema.read_schema(path)
data = FileDataStream(path, schema)

# CV also accepts the list of pipeline steps directly as input
pipeline_steps = [
    Indicator() << {
        'Ozone_ind': 'Ozone',
        'Solar_R_ind': 'Solar_R'},
    Handler(
        replace_with='Mean') << {
        'Solar_R': 'Solar_R',
        'Ozone': 'Ozone'},
    FastLinearRegressor(
        feature=[
            'Ozone',
            'Solar_R',
            'Ozone_ind',
            'Solar_R_ind',
            'Temp'],
        label='Wind')]

# Since the Indicator and Handler transforms don't learn from data,
# they could be run once before splitting the data into folds, instead of
# repeating them once per fold. We use 'split_start=after_transforms' option
# to achieve this optimization.
cv_results = CV(pipeline_steps).fit(data, split_start='after_transforms')

# Results can be accessed the same way as in Case 1 above.
print(cv_results['metrics_summary'])
