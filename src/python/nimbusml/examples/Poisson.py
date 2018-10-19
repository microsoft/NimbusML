###############################################################################
# Poisson Loss
from nimbusml.linear_model import OnlineGradientDescentRegressor
from nimbusml.loss import Poisson

# specifying the loss function as a string keyword
trainer1 = OnlineGradientDescentRegressor(loss='poisson')

# can also use loss class instead of string

trainer2 = OnlineGradientDescentRegressor(
    loss=Poisson())  # equivalent to loss='tweedie'
