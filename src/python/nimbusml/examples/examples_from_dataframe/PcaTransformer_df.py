###############################################################################
# PcaTransformer
from nimbusml import Pipeline, FileDataStream
from nimbusml.datasets import get_dataset
from nimbusml.decomposition import PcaTransformer
from nimbusml.ensemble import LightGbmBinaryClassifier
from nimbusml.preprocessing.schema import ColumnConcatenator

# use the built-in data set 'infert' to create test and train data
#   Unnamed: 0  education   age  parity  induced  case  spontaneous  stratum  \
# 0           1        0.0  26.0     6.0      1.0   1.0          2.0      1.0
# 1           2        0.0  42.0     1.0      1.0   1.0          0.0      2.0
#   pooled.stratum education_str
# 0             3.0        0-5yrs
# 1             1.0        0-5yrs

train_file = get_dataset("infert").as_filepath()
schema = "col=none:R4:0 col=education:R4:1 col=age:R4:2 col=parity:R4:3 " \
         "col=induced:R4:4 col=case:R4:5 col=spontaneous:R4:6 " \
         "col=stratum:R4:7 col=pooledstratum:R4:8 col=educationstr:R4:9 " \
         "sep=, header=+"
fds = FileDataStream(train_file, schema=schema)

# target and features columns
y = 'case'
X = ['age', 'parity', 'induced', 'spontaneous', 'stratum', 'pooledstratum']

# observe gradual impact of dimensionality reduction on AUC
# reducing dimensions should degrade signal gradually, while
# maintaining the traits in original dataset as much as possible.
for rank in range(len(X), 2, -1):
    print('Number of dimensions=', rank)
    pipe = Pipeline([
        ColumnConcatenator() << {'X': X},  # X is VectorDataViewType column
        PcaTransformer(rank=rank) << 'X',  # find principal components of X
        LightGbmBinaryClassifier()
    ])
    metrics, scores = pipe.fit(fds, y).test(fds, y)
    print('AUC=', metrics['AUC'].values)
