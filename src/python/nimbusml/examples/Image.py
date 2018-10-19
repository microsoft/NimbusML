###############################################################################
# Loader
# Resizer
# PixelExtractor
# DnnFeaturizer
# FastLinearBinaryClassifier
import numpy as np
import pandas
from nimbusml import Pipeline
from nimbusml.datasets.image import get_RevolutionAnalyticslogo, get_Microsoftlogo
from nimbusml.feature_extraction.image import Loader, Resizer, PixelExtractor
from nimbusml.linear_model import FastLinearBinaryClassifier

data = pandas.DataFrame(data=dict(
    Path=[get_RevolutionAnalyticslogo(), get_Microsoftlogo()],
    Label=[True, False]))

X = data[['Path']]
y = data[['Label']]

# define the training pipeline
pipeline = Pipeline([
    Loader(columns={'ImgPath': 'Path'}),
    Resizer(image_width=227, image_height=227,
            columns={'ImgResize': 'ImgPath'}),
    PixelExtractor(columns={'ImgPixels': 'ImgResize'}),
    FastLinearBinaryClassifier(feature='ImgPixels')
])

# train
pipeline.fit(X, y)

# predict
nimbusml_pred = pipeline.predict(X)
print("Predicted Labels : {0}".format(nimbusml_pred.PredictedLabel.values))
# Predicted Labels : [True False]
print(
    "Accuracy : {0}".format(np.mean(
        y.Label.values == nimbusml_pred.PredictedLabel.values)))
# Accuracy : 1
