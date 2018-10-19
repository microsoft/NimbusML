###############################################################################
# KMeansPlusPlus
import pandas
from nimbusml import Pipeline
from nimbusml.cluster import KMeansPlusPlus

# define 3 clusters with centroids (1,1,1), (11,11,11) and (-11,-11,-11)
X_train = pandas.DataFrame(data=dict(
    x=[0, 1, 2, 10, 11, 12, -10, -11, -12],
    y=[0, 1, 2, 10, 11, 12, -10, -11, -12],
    z=[0, 1, 2, 10, 11, 12, -10, -11, -12]))

# these should clearly belong to just 1 of the 3 clusters
X_test = pandas.DataFrame(data=dict(
    x=[-1, 3, 9, 13, -13, -20],
    y=[-1, 3, 9, 13, -13, -20],
    z=[-1, 3, 9, 13, -13, -20]))

y_test = pandas.DataFrame(data=dict(
    clusterid=[2, 2, 1, 1, 0, 0]))

pipe = Pipeline([
    KMeansPlusPlus(n_clusters=3)
]).fit(X_train)

metrics, predictions = pipe.test(X_test, y_test, output_scores=True)

# print predictions
print(predictions.head())

# print evaluation metrics
print(metrics)
