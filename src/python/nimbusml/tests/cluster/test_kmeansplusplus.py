# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import unittest

import pandas
from nimbusml.cluster import KMeansPlusPlus
from sklearn.utils.testing import assert_equal, assert_not_equal


class TestKMeansPlusPlus(unittest.TestCase):

    def test_kmeansplusplus(self):
        X_train = pandas.DataFrame(data=dict(
            x=[0, 1, 2, 10, 11, 12, -10, -11, -12],
            y=[0, 1, 2, 10, 11, 12, -10, -11, -12],
            z=[0, 1, 2, 10, 11, 12, -10, -11, -12]))

        # these should clearly belong to just 1 of the 3 clusters
        X_test = pandas.DataFrame(data=dict(
            x=[-1, 3, 9, 13, -13, -20],
            y=[-1, 3, 9, 13, -13, -20],
            z=[-1, 3, 9, 13, -13, -20]))

        mymodel = KMeansPlusPlus(n_clusters=3).fit(X_train)
        scores = mymodel.predict(X_test)

        # Evaluate the model
        # currently KMeansPlusPlus doesnt return label values but its internal
        # label enumeration
        # which could be random so accuracy cant be measure like below
        # accuracy = np.mean(y_test == [i for i in scores])
        # assert_equal(accuracy[0], 1.0, "accuracy should be %s" % 1.0)

        # we can measure instead that elements are grouped into 3 clusters
        assert_equal(
            scores[0],
            scores[1],
            "elements should be in the same cluster")
        assert_not_equal(
            scores[1],
            scores[2],
            "elements should be in different clusters")
        assert_equal(
            scores[2],
            scores[3],
            "elements should be in the same cluster")
        assert_not_equal(
            scores[3],
            scores[4],
            "elements should be in different clusters")
        assert_equal(
            scores[4],
            scores[5],
            "elements should be in the same cluster")

    def test_kmeansplusplus_noy(self):
        X_train = pandas.DataFrame(data=dict(
            x=[0, 1, 2, 10, 11, 12, -10, -11, -12],
            y=[0, 1, 2, 10, 11, 12, -10, -11, -12],
            z=[0, 1, 2, 10, 11, 12, -10, -11, -12]))

        # these should clearly belong to just 1 of the 3 clusters
        X_test = pandas.DataFrame(data=dict(
            x=[-1, 3, 9, 13, -13, -20],
            y=[-1, 3, 9, 13, -13, -20],
            z=[-1, 3, 9, 13, -13, -20]))

        mymodel = KMeansPlusPlus(n_clusters=3).fit(
            X_train)  # does not work without y_train
        scores = mymodel.predict(X_test)

        # Evaluate the model
        # currently KMeansPlusPlus doesnt return label values but its internal
        # label enumeration
        # which could be random so accuracy cant be measure like below
        # accuracy = np.mean(y_test == [i for i in scores])
        # assert_equal(accuracy[0], 1.0, "accuracy should be %s" % 1.0)

        # we can measure instead that elements are grouped into 3 clusters
        assert_equal(
            scores[0],
            scores[1],
            "elements should be in the same cluster")
        assert_not_equal(
            scores[1],
            scores[2],
            "elements should be in different clusters")
        assert_equal(
            scores[2],
            scores[3],
            "elements should be in the same cluster")
        assert_not_equal(
            scores[3],
            scores[4],
            "elements should be in different clusters")
        assert_equal(
            scores[4],
            scores[5],
            "elements should be in the same cluster")


if __name__ == '__main__':
    unittest.main()
