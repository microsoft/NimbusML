###############################################################################
# Squared Loss
from nimbusml.linear_model import FastLinearRegressor
# can also use loss class instead of string
from nimbusml.loss import Squared

# specifying the loss function as a string keyword
trainer1 = FastLinearRegressor(loss='squared')

trainer2 = FastLinearRegressor(loss=Squared())  # equivalent to loss='squared'
