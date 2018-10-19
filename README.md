# NimbusML

`nimbusml` is a Python module that provides experimental Python bindings for [ML.NET](https://github.com/dotnet/machinelearning). 

ML.NET was originally developed in Microsoft Research and is used across many product groups in Microsoft like Windows, Bing, PowerPoint, Excel and others. `nimbusml` was built to enable data science teams that are more familiar with Python to take advantage of ML.NET's functionality and performance. 

This package enables training ML.NET pipelines or integrating ML.NET components directly into Scikit-Learn pipelines (it supports  `numpy.ndarray`, `scipy.sparse_cst`, and `pandas.DataFrame` as inputs).

Documentation can be found [here](https://docs.microsoft.com/en-us/NimbusML/overview) with additional [notebook samples](https://github.com/Microsoft/NimbusML-Samples).

## Installation

`nimbusml` runs on Windows, Linux, and macOS - any platform where 64 bit .NET Core is available. It relies on .NET Core, and this is installed automatically as part of the package.

`nimbusml` requires Python 2.7, 3.5, or 3.6, 64 bit version only.

Install `nimbusml` using `pip` with:

```
pip install nimbusml
```

`nimbusml` has been tested on Windows 10, MacOS 10.13, Ubuntu 14.04, Ubuntu 16.04, Ubuntu 18.04, CentOS 7, and RHEL 7.

## Examples

Here is an example of how to train a model to predict sentiment from text samples (based on the ML.NET example [here](https://github.com/dotnet/machinelearning/blob/master/README.md))

```python
pipeline = Pipeline([ # nimbusml pipeline
    NGramFeaturizer(columns={'Features': ['SentimentText']}),
	FastTreeBinaryClassifier(feature=['Features'], 
	                                   label='Sentiment')
])

# fit and predict
pipeline.fit(data)
results = pipeline.predict(data)
```

Instead of creating an `nimbusml` pipeline, you can also integrate components into Scikit-Learn pipelines:

```python
pipeline = Pipeline([ # sklearn pipeline
    ('tfidf', TfidfVectorizer()), # sklearn transform
    ('clf', FastTreeBinaryClassifier())]) # nimbusml learner
])

# fit and predict
pipeline.fit(data)
results = pipeline.predict(data)
```



Many additional examples and tutorials can be found in the [documentation](https://docs.microsoft.com/en-us/NimbusML/overview).


## Building

To build `nimbusml` from source please visit our [developers guide](docs/developers/developer-guide.md).

## Contributing

We welcome [contributions](docs/project-docs/contributing.md)!

## License

NimbusML is licensed under the [MIT license](LICENSE).

