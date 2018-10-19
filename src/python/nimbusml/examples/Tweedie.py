###############################################################################
# Tweedie Loss
from nimbusml.linear_model import OnlineGradientDescentRegressor
# can also use loss class instead of string
from nimbusml.loss import Tweedie

# specifying the loss function as a string keyword
trainer1 = OnlineGradientDescentRegressor(loss='tweedie')

trainer2 = OnlineGradientDescentRegressor(
    loss=Tweedie())  # equivalent to loss='tweedie'
trainer3 = OnlineGradientDescentRegressor(loss=Tweedie(index=3.0))
