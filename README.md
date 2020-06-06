# NimbusML

`nimbusml` is a Python module that provides Python bindings for [ML.NET](https://github.com/dotnet/machinelearning). 

ML.NET was originally developed in Microsoft Research and is used across many product groups in Microsoft like Windows, Bing, PowerPoint, Excel, and others. `nimbusml` was built to enable data science teams that are more familiar with Python to take advantage of ML.NET's functionality and performance. 

`nimbusml` enables training ML.NET pipelines or integrating ML.NET components directly into [scikit-learn](https://scikit-learn.org/stable/) pipelines. It adheres to existing `scikit-learn` conventions, allowing simple interoperability between `nimbusml` and `scikit-learn` components, while adding a suite of fast, highly optimized, and scalable algorithms, transforms, and components written in C++ and C\#.

See examples below showing interoperability with `scikit-learn`. A more detailed example in the [documentation](https://docs.microsoft.com/en-us/nimbusml/tutorials/b_c-sentiment-analysis-3-combining-nimbusml-and-scikit-learn) shows how to use a `nimbusml` component in a `scikit-learn` pipeline, and create a pipeline using only `nimbusml` components.

`nimbusml` supports `numpy.ndarray`, `scipy.sparse_cst`, and `pandas.DataFrame` as inputs. In addition, `nimbusml` also supports streaming from files without loading the dataset into memory with `FileDataStream`, which allows training on data significantly exceeding memory.

Documentation can be found [here](https://docs.microsoft.com/en-us/NimbusML/overview) and additional notebook samples can be found [here](https://github.com/Microsoft/NimbusML-Samples).

## Installation

`nimbusml` runs on Windows, Linux, and macOS. 

`nimbusml` requires Python **2.7**, **3.5**, **3.6**, **3.7** 64 bit version only.

Install `nimbusml` using `pip` with:

```
pip install nimbusml
```

`nimbusml` has been reported to work on Windows 10, MacOS 10.13, Ubuntu 14.04, Ubuntu 16.04, Ubuntu 18.04, CentOS 7, and RHEL 7.

## Examples

Here is an example of how to train a model to predict sentiment from text samples (based on [this](https://github.com/dotnet/machinelearning/blob/master/README.md) ML.NET example). The full code for this example is [here](https://github.com/Microsoft/NimbusML-Samples/blob/master/samples/2.1%20%5BText%5D%20Sentiment%20Analysis%201%20-%20Data%20Loading%20with%20Pandas.ipynb).

```python
from nimbusml import Pipeline, FileDataStream
from nimbusml.datasets import get_dataset
from nimbusml.ensemble import FastTreesBinaryClassifier
from nimbusml.feature_extraction.text import NGramFeaturizer

train_file = get_dataset('gen_twittertrain').as_filepath()
test_file = get_dataset('gen_twittertest').as_filepath()

train_data = FileDataStream.read_csv(train_file, sep='\t')
test_data = FileDataStream.read_csv(test_file, sep='\t')

pipeline = Pipeline([ # nimbusml pipeline
    NGramFeaturizer(columns={'Features': ['Text']}),
    FastTreesBinaryClassifier(feature=['Features'], label='Label')
])

# fit and predict
pipeline.fit(train_data)
results = pipeline.predict(test_data)
```

Instead of creating an `nimbusml` pipeline, you can also integrate components into scikit-learn pipelines:

```python
from sklearn.pipeline import Pipeline
from nimbusml.datasets import get_dataset
from nimbusml.ensemble import FastTreesBinaryClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

train_file = get_dataset('gen_twittertrain').as_filepath()
test_file = get_dataset('gen_twittertest').as_filepath()

train_data = pd.read_csv(train_file, sep='\t')
test_data = pd.read_csv(test_file, sep='\t')

pipeline = Pipeline([ # sklearn pipeline
    ('tfidf', TfidfVectorizer()), # sklearn transform
    ('clf', FastTreesBinaryClassifier()) # nimbusml learner
])

# fit and predict
pipeline.fit(train_data["Text"], train_data["Label"])
results = pipeline.predict(test_data["Text"])
```



Many additional examples and tutorials can be found in the [documentation](https://docs.microsoft.com/en-us/NimbusML/overview).


## Building

To build `nimbusml` from source please visit our [developer guide](docs/developers/developer-guide.md).

## Contributing

The contributions guide can be found [here](CONTRIBUTING.md). 

## Support

If you have an idea for a new feature or encounter a problem, please open an [issue](https://github.com/Microsoft/NimbusML/issues/new) in this repository or ask your question on Stack Overflow.

## License

NimbusML is licensed under the [MIT license](LICENSE).

