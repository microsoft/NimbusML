###############################################################################
# ToKeyImputer

import pandas
from nimbusml.preprocessing import ToKeyImputer

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
            None,
            "spider"]))

tokey = ToKeyImputer() << 'text'
y = tokey.fit_transform(text_df)
print(y)

#          text
# 0         cat
# 1         dog
# 2        fish
# 3      orange
# 4  cat orange
# 5         dog
# 6        fish
# 7         dog   <== Missing value has been replaced
# 8      spider
