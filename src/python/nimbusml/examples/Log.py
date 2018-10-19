###############################################################################
# Log Loss
from nimbusml.linear_model import FastLinearBinaryClassifier
from nimbusml.loss import Log

# specifying the loss function as a string keyword
trainer1 = FastLinearBinaryClassifier(loss='log')

# can also use loss class instead of string

trainer1 = FastLinearBinaryClassifier(loss=Log())  # equivalent to loss='log'
