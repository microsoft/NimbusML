###############################################################################
# Exponential Loss
from nimbusml.linear_model import SgdBinaryClassifier
from nimbusml.loss import Exp

# specify loss function using string keyword
trainer1 = SgdBinaryClassifier(loss='exp')

# can also use the loss class instead of string.

trainer1 = SgdBinaryClassifier(loss=Exp())  # equivalent to loss='exp'
trainer2 = SgdBinaryClassifier(loss=Exp(beta=0.4))
