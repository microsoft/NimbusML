# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import unittest

from nimbusml import DprepDataStream

class TestDprep(unittest.TestCase):

    def test_dprep_to_df(self):
        dprep = DprepDataStream("C:/tmp/a.dprep")
        transformed_data_as_df = dprep.to_df()

if __name__ == '__main__':
    unittest.main()
