###############################################################################
# Smoothed Hinge Loss
from nimbusml.linear_model import FastLinearBinaryClassifier
# can also use loss class instead of string
from nimbusml.loss import SmoothedHinge

# specifying the loss function as a string keyword
trainer1 = FastLinearBinaryClassifier(loss='smoothed_hinge')

# equivalent to loss='smoothed_hinge'
trainer2 = FastLinearBinaryClassifier(loss=SmoothedHinge())
trainer3 = FastLinearBinaryClassifier(loss=SmoothedHinge(smoothing_const=0.5))
