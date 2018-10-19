NimbusML

`nimbusml` provides battle-tested state-of-the-art ML algorithms,
transforms and components, aiming to make them useful for all
developers, data scientists, and information workers and helpful in all
products, services and devices. The components are authored by the team
members, as well as numerous contributors from MSR, CISL, Bing and other
teams at Microsoft.

`nimbusml` is interoperable with `scikit-learn` estimators and transforms,
while adding a suite of highly optimized algorithms written in C++ and
C\# for speed and performance. `nimbusml` trainers and transforms support
the following data structures for the `fit()` and `transform()` methods:

-   `numpy.ndarray`
-   `scipy.sparse_cst`
-   `pandas.DataFrame`.

In addition, `nimbusml` also supports streaming from files without loading
the dataset into memory, which allows training on data significantly
exceeding memory using `FileDataStream`.

With `FileDataStream` `nimbusml` is able to handle up to **billion** features
 and **billions** of training examples for select algorithms.

For more details, please refer to the documentation:
<https://docs.microsoft.com/en-us/nimbusml>.