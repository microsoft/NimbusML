###############################################################################
# PCAAnomalyDetector
import numpy as np
import pandas
from nimbusml.datasets import get_dataset
from nimbusml.decomposition import PcaAnomalyDetector
from sklearn.model_selection import train_test_split

# use 'iris' data set to create test and train data
#    Sepal_Length  Sepal_Width  Petal_Length  Petal_Width Label Species  Setosa
# 0           5.1          3.5           1.4          0.2     0  setosa     1.0
# 1           4.9          3.0           1.4          0.2     0  setosa     1.0

df = get_dataset("iris").as_df()
df.drop(['Label', 'Setosa', 'Species'], axis=1, inplace=True)

X_train, X_test = train_test_split(df)

# homogenous values for labels, y-column required by scikit
y_train = np.ones(len(X_train))

svm = PcaAnomalyDetector(rank=3)
svm.fit(X_train)

# add additional non-iris data to the test data set
not_iris = pandas.DataFrame(data=dict(Sepal_Length=[2.5, 2.6],
                                      Sepal_Width=[.75, .9],
                                      Petal_Length=[2.5, 2.5],
                                      Petal_Width=[.8, .7]))

merged_test = pandas.concat([X_test, not_iris], sort=False)

scores = svm.predict(merged_test)

# look at the last few observations
print(scores.tail())
