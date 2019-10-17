# NimbusML

`nimbusml` is a Python module that provides Python bindings for [ML.NET](https://github.com/dotnet/machinelearning). 

`nimbusml` aims to enable data science teams that are more familiar with Python
to take advantage of ML.NET's functionality and performance. It provides
battle-tested state-of-the-art ML algorithms, transforms, and components. The
components are authored by the team members, as well as numerous contributors
from MSR, CISL, Bing and other teams at Microsoft.

`nimbusml` is interoperable with `scikit-learn` estimators and transforms,
while adding a suite of fast, highly optimized, and scalable algorithms written
in C++ and C\#. `nimbusml` trainers and transforms support the following data
structures for the `fit()` and `transform()` methods:

-   `numpy.ndarray`
-   `scipy.sparse_cst`
-   `pandas.DataFrame`.

In addition, `nimbusml` also supports streaming from files without loading the
dataset into memory with `FileDataStream`, which allows training on data
significantly exceeding memory.

With `FileDataStream`, `nimbusml` is able to handle up to a **billion**
features and **billions** of training examples for select algorithms.

For more details, please refer to the documentation:
<https://docs.microsoft.com/en-us/nimbusml>.

## Third party notices

`nimbusml` contains ML.NET binaries and the .NET Core CLR runtime, as well as
their dependencies. Both ML.NET and .NET Core CLR are made available under the
MIT license. Please refer to the [third party notices](https://github.com/microsoft/NimbusML/blob/master/THIRD-PARTY-NOTICES.txt)
for full licensing information for ML.NET and .NET Core CLR.