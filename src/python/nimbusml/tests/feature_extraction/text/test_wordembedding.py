# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import os
import platform
import unittest

import pandas as pd
from nimbusml import FileDataStream, Pipeline
from nimbusml.datasets import get_dataset
from nimbusml.feature_extraction.text import NGramFeaturizer
from nimbusml.feature_extraction.text import WordEmbedding
from nimbusml.feature_extraction.text.extractor import Ngram
from nimbusml.preprocessing.schema import ColumnConcatenator
from sklearn.utils.testing import assert_almost_equal


class TestWordEmbedding(unittest.TestCase):

    # TODO: fix ssl issue on test centos7 & ubuntu14 boxes.
    # Test works on ubuntu16.
    # Currently centos7/ubuntu14 boxes give this error:
    # Error: l3g.txt: Could not download. WebClient returned the following
    # error: An error occurred while sending the request.
    # Problem with the SSL CA cert (path? access rights?)
    # Error: AdSelectionFullyConnect.zip: Could not download. WebClient
    # returned the following error: An error occurred while sending the
    # request. Problem with the SSL CA cert (path? access rights?)
    # Error: *** System.InvalidOperationException: 'Error downloading resource
    # from
    # 'https://aka.ms/tlc-resources/text/dssm/AdSelectionFullyConnect.zip': An
    # error occurred while sending the request. Problem with the SSL CA cert
    # (path? access rights?)
    @unittest.skipIf(
        os.name != "nt" and (
            platform.linux_distribution()[0] != "Ubuntu" or
            platform.linux_distribution()[1] != "16.04"),
        "not supported on this platform")
    def test_ssweembedding(self):
        wordvectors = pd.DataFrame(data=dict(w1=["like", "hate", "okay"],
                                             w2=["great", "horrible",
                                                 "lukewarm"],
                                             w3=["awesome", "worst",
                                                 "boring"]))
        mycols = ['w1', 'w2', 'w3']
        concat = ColumnConcatenator() << {'features': mycols}
        sswe = WordEmbedding() << 'features'
        pipeline = Pipeline([concat, sswe])
        y = pipeline.fit_transform(wordvectors)
        y = y[[col for col in y.columns if 'features' in col]]
        assert_almost_equal(y.sum().sum(), -97.6836, decimal=4,
                            err_msg="Sum should be %s" % -97.6836)

    # TODO: fix ssl issue on test centos7 & ubuntu14 boxes.
    # Test works on ubuntu16.
    # Currently centos7/ubuntu14 boxes give this error:
    # Error: l3g.txt: Could not download. WebClient returned the following
    # error: An error occurred while sending the request.
    # Problem with the SSL CA cert (path? access rights?)
    # Error: AdSelectionFullyConnect.zip: Could not download. WebClient
    # returned the following error: An error occurred while sending the
    # request. Problem with the SSL CA cert (path? access rights?)
    # Error: *** System.InvalidOperationException: 'Error downloading resource
    # from
    # 'https://aka.ms/tlc-resources/text/dssm/AdSelectionFullyConnect.zip': An
    # error occurred while sending the request. Problem with the SSL CA cert
    # (path? access rights?)
    @unittest.skipIf(
        os.name != "nt" and (
            platform.linux_distribution()[0] != "Ubuntu" or
            platform.linux_distribution()[1] != "16.04"),
        "not supported on this platform")
    def test_word_embedding_example(self):
        path = get_dataset('infert').as_filepath()
        file_schema = 'sep=, col=id:TX:0 col=education:TX:1 col=age:R4:2 ' \
                      'col=parity:R4:3 col=induced:R4:4 col=case:R4:5 ' \
                      'col=spontaneous:R4:6 quote+ header=+'
        data = FileDataStream(path, schema=file_schema)

        # transform usage
        # TODO: Bug 146763
        # TODO: Bug 149666
        # TODO: Bug 149700
        pipeline = Pipeline([
            NGramFeaturizer(word_feature_extractor=Ngram(),
                            output_tokens_column_name='features_TransformedText',
                            columns={'features': ['id', 'education']}),

            WordEmbedding(columns='features_TransformedText')
        ])

        features = pipeline.fit_transform(data)
        assert features.shape == (248, 802)

    # TODO: fix ssl issue on test centos7 & ubuntu14 boxes.
    # Test works on ubuntu16.
    # Currently centos7/ubuntu14 boxes give this error:
    # Error: l3g.txt: Could not download. WebClient returned the following
    # error: An error occurred while sending the request.
    # Problem with the SSL CA cert (path? access rights?)
    # Error: AdSelectionFullyConnect.zip: Could not download. WebClient
    # returned the following error: An error occurred while sending the
    # request. Problem with the SSL CA cert (path? access rights?)
    # Error: *** System.InvalidOperationException: 'Error downloading resource
    # from
    # 'https://aka.ms/tlc-resources/text/dssm/AdSelectionFullyConnect.zip': An
    # error occurred while sending the request. Problem with the SSL CA cert
    # (path? access rights?)
    @unittest.skipIf(
        os.name != "nt" and (
            platform.linux_distribution()[0] != "Ubuntu" or
            platform.linux_distribution()[1] != "16.04"),
        "not supported on this platform")
    def test_word_embedding_example2(self):
        path = get_dataset('infert').as_filepath()
        file_schema = 'sep=, col=id:TX:0 col=education:TX:1 col=age:R4:2 ' \
                      'col=parity:R4:3 col=induced:R4:4 col=case:R4:5 ' \
                      'col=spontaneous:R4:6 quote+ header=+'
        data = FileDataStream(path, schema=file_schema)

        pipeline = Pipeline([
            NGramFeaturizer(word_feature_extractor=Ngram(),
                            output_tokens_column_name='features_TransformedText',
                            columns={'features': ['id', 'education']}),

            WordEmbedding(columns='features_TransformedText')
        ])

        features = pipeline.fit_transform(data)
        assert features.shape == (248, 802)
        assert 'features_TransformedText.94' in list(features.columns)

    # TODO: fix ssl issue on test centos7 & ubuntu14 boxes.
    # Test works on ubuntu16.
    # Currently centos7/ubuntu14 boxes give this error:
    # Error: l3g.txt: Could not download. WebClient returned the following
    # error: An error occurred while sending the request.
    # Problem with the SSL CA cert (path? access rights?)
    # Error: AdSelectionFullyConnect.zip: Could not download. WebClient
    # returned the following error: An error occurred while sending the
    # request. Problem with the SSL CA cert (path? access rights?)
    # Error: *** System.InvalidOperationException: 'Error downloading resource
    # from
    # 'https://aka.ms/tlc-resources/text/dssm/AdSelectionFullyConnect.zip': An
    # error occurred while sending the request. Problem with the SSL CA cert
    # (path? access rights?)
    @unittest.skipIf(
        os.name != "nt" and (
            platform.linux_distribution()[0] != "Ubuntu" or
            platform.linux_distribution()[1] != "16.04"),
        "not supported on this platform")
    def test_word_embedding_example_dict_same_name(self):
        path = get_dataset('infert').as_filepath()
        file_schema = 'sep=, col=id:TX:0 col=education:TX:1 col=age:R4:2 ' \
                      'col=parity:R4:3 col=induced:R4:4 col=case:R4:5 ' \
                      'col=spontaneous:R4:6 quote+ header=+'
        data = FileDataStream(path, schema=file_schema)
        pipeline = Pipeline([
            NGramFeaturizer(word_feature_extractor=Ngram(), output_tokens_column_name='features_TransformedText',
                            columns={'features': ['id', 'education']}),

            # What is features_TransformedText?
            WordEmbedding(
                columns={
                    'features_TransformedText': 'features_TransformedText'})
        ])

        features = pipeline.fit_transform(data)
        assert features.shape == (248, 802)

    @unittest.skip('System.ArgumentOutOfRangeException')
    def test_word_embedding_example_dict_newname(self):
        path = get_dataset('infert').as_filepath()
        file_schema = 'sep=, col=id:TX:0 col=education:TX:1 col=age:R4:2 ' \
                      'col=parity:R4:3 col=induced:R4:4 col=case:R4:5 ' \
                      'col=spontaneous:R4:6 quote+ header=+'
        data = FileDataStream(path, schema=file_schema)
        pipeline = Pipeline([
            NGramFeaturizer(word_feature_extractor=Ngram(),
                            output_tokens_column_name='features_TransformedText',
                            columns={'features': ['id', 'education']}),

            # What is features_TransformedText?
            WordEmbedding(
                columns={
                    'features_TransformedText2': 'features_TransformedText'})
        ])

        features = pipeline.fit_transform(data)
        assert features.shape == (248, 409)


if __name__ == '__main__':
    unittest.main()
