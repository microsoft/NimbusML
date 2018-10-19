###############################################################################
# ToKey

import pandas
from nimbusml.preprocessing import ToKey

# Create the data
text_df = pandas.DataFrame(
    data=dict(
        text=[
            "cat",
            "dog",
            "fish",
            "orange",
            "cat orange",
            "dog",
            "fish",
            "spider"]))

tokey = ToKey() << 'text'
y = tokey.fit_transform(text_df)
print(y)
