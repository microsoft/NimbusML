###############################################################################
# OrdinaryLeastSquaresRegressor
from nimbusml import Pipeline, FileDataStream, Role
from nimbusml.datasets import get_dataset
from nimbusml.linear_model import OrdinaryLeastSquaresRegressor
from nimbusml.preprocessing.missing_values import Filter

# use the built-in data set 'airquality' to create test and train data
#    Unnamed: 0  Ozone  Solar_R  Wind  Temp  Month  Day
# 0           1   41.0    190.0   7.4    67      5    1
# 1           2   36.0    118.0   8.0    72      5    2

train_file = get_dataset("airquality").as_filepath()
schema = "col=none:R4:0 col=ozone:R4:1 col=solar:R4:2 col=wind:R4:3 " \
         "col=temp:R4:4 col=month:R4:5 col=day:R4:6 sep=, header=+"

fds = FileDataStream(train_file, schema=schema)

# set up pipeline
pipe = Pipeline([Filter() << ['ozone'], OrdinaryLeastSquaresRegressor() << {
    Role.Label: 'ozone',
    Role.Feature: ['solar', 'wind', 'temp', 'month', 'day']}])

# train and evaluate the model
metrics, scores = pipe.fit(fds).test(fds, "ozone", output_scores=True)
print(metrics)
