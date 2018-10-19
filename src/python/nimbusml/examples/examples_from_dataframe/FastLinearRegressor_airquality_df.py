###############################################################################
# FastLinearRegressor
import numpy as np
from nimbusml.datasets import get_dataset
from nimbusml.linear_model import FastLinearRegressor
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

# train a model and score
flinear = FastLinearRegressor().fit(X_train, y_train)
scores = flinear.predict(X_test)

# evaluate the model
print('R-squared fit:', r2_score(y_test, scores, ))
