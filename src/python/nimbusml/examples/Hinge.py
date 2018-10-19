###############################################################################
# Hinge Loss
from nimbusml.linear_model import AveragedPerceptronBinaryClassifier
from nimbusml.loss import Hinge

# specify the loss function as a string keyword
trainer1 = AveragedPerceptronBinaryClassifier(loss='hinge')

# can also use the loss class instead of string

trainer1 = AveragedPerceptronBinaryClassifier(
    loss=Hinge())  # equivalent to loss='hinge'
trainer2 = AveragedPerceptronBinaryClassifier(loss=Hinge(margin=2.0))
