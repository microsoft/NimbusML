###############################################################################
# Loader, Resizer, PixelExtractor, DnnFeaturizer
import numpy as np
import pandas
from nimbusml import Pipeline
from nimbusml.datasets.image import get_RevolutionAnalyticslogo, get_Microsoftlogo
from nimbusml.feature_extraction.image import Loader
from nimbusml.feature_extraction.image import PixelExtractor
from nimbusml.feature_extraction.image import Resizer
from nimbusml.linear_model import FastLinearBinaryClassifier

data = pandas.DataFrame(data=dict(
    Path=[get_RevolutionAnalyticslogo(), get_Microsoftlogo()],
    Label=[True, False]))

X = data[['Path']]
y = data[['Label']]

# transforms and learners
transform_1 = Loader() << 'Path'
transform_2 = Resizer(image_width=227, image_height=227)
transform_3 = PixelExtractor()
algo = FastLinearBinaryClassifier() << 'Path'

# pipeline of transforms and trainer
pipeline = Pipeline([transform_1, transform_2, transform_3, algo])

# fit the model
pipeline.fit(X, y)

# scoring
scores = pipeline.predict(X)
print("Predicted Labels:", scores.PredictedLabel.values)
print("Accuracy:", np.mean(y.Label.values == scores.PredictedLabel.values))
