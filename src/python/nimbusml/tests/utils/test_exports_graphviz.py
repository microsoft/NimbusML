# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
import os
import tempfile
import unittest
import warnings

import pandas
from nimbusml import Pipeline
from nimbusml.feature_extraction.categorical import OneHotVectorizer
from nimbusml.internal.utils.data_roles import Role
from nimbusml.linear_model import FastLinearRegressor
from nimbusml.preprocessing.normalization import MeanVarianceScaler
from nimbusml.preprocessing.schema import ColumnDropper as Drop
from nimbusml.utils.exports import img_export_pipeline

_sentiments = """
"ItemID"	"Sentiment"	"SentimentSource"	"SentimentText"	"RowNum"	\
"Positive"	"Train"	"Small"
1	0	"Sentiment140"	"is so sad for my APL friend............."	1	\
FALSE	TRUE	FALSE
2	0	"Sentiment140"	"I missed the New Moon trailer..."	2	FALSE	\
TRUE	FALSE
3	1	"Sentiment140"	"omg its already 7:30 :O"	3	TRUE	TRUE	\
FALSE
4	0	"Sentiment140"	".. Omgaga. Im sooo  im gunna CRy. I've been at this \
dentist since 11.. I was suposed 2 just get a crown put on (30mins)..."	4	\
FALSE	TRUE	FALSE
5	0	"Sentiment140"	"i think mi bf is cheating on me!!!       T_T"	5	\
FALSE	TRUE	FALSE
6	0	"Sentiment140"	"or i just worry too much?"	6	FALSE	TRUE	\
FALSE
7	1	"Sentiment140"	"Juuuuuuuuuuuuuuuuussssst Chillin!!"	7	TRUE	\
TRUE	FALSE
8	0	"Sentiment140"	"Sunny Again        Work Tomorrow  :-|       \
TV Tonight"	8	FALSE	TRUE	FALSE
9	1	"Sentiment140"	"handed in my uniform today . i miss you already"	\
9	TRUE	TRUE	FALSE
"""


class TestExportsGraphviz(unittest.TestCase):

    def test_pipeline_exports(self):
        import graphviz.backend
        df = pandas.DataFrame(dict(education=['A', 'B', 'A', 'B', 'A'],
                                   workclass=['X', 'X', 'Y', 'Y', 'Y'],
                                   yy=[1.1, 2.2, 1.24, 3.4, 3.4]))

        exp = Pipeline([
            MeanVarianceScaler() << {'new_y': 'yy'},
            OneHotVectorizer() << ['workclass', 'education'],
            Drop() << 'yy',
            FastLinearRegressor() << {'Feature': ['workclass', 'education'],
                                      Role.Label: 'new_y'}
        ])

        gr = img_export_pipeline(exp, df)
        name = next(tempfile._get_candidate_names())
        try:
            gr.render(name)
            assert os.path.exists(name)
        except graphviz.backend.ExecutableNotFound:
            warnings.warn('Graphviz is not installed.')
        if os.path.exists(name):
            os.remove(name)


if __name__ == "__main__":
    unittest.main()
