# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import unittest

from nimbusml import FileDataStream
from nimbusml.datasets import get_dataset
from nimbusml.feature_extraction.categorical import OneHotVectorizer

try:
    from pandas.testing import assert_frame_equal
except ImportError:
    # earlier versions
    from pandas.util.testing import assert_frame_equal


class TestOneHotVectorizer(unittest.TestCase):

    def test_fit_transform(self):
        # data input (as a FileDataStream)
        path = get_dataset('infert').as_filepath()

        data = FileDataStream.read_csv(path)

        # transform usage
        xf = OneHotVectorizer(
            columns={
                'edu': 'education',
                'in': 'induced',
                'sp': 'spontaneous'})

        # fit and transform
        res1 = xf.fit_transform(data)
        res2 = xf.fit(data).transform(data)
        assert_frame_equal(res1, res2)


if __name__ == '__main__':
    unittest.main()
