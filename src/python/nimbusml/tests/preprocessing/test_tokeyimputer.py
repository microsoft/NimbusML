# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import unittest

import numpy as np
import pandas as pd
from nimbusml.preprocessing import ToKeyImputer


class TestToKeyImputer(unittest.TestCase):

    def test_tokeyimputer(self):
        text_df = pd.DataFrame(
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

        self.assertEqual(y.loc[7, 'text'], 'dog')


if __name__ == '__main__':
    unittest.main()
