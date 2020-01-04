.. rxtitle:: Data Sources
.. rxdescription:: Accepted data source for nimbusml training

.. _datasources:

============
Data Sources
============

.. index:: column type

.. contents::
    :local:

Input Data Types for Transforms and Trainers
----------------------------------------------

The transforms and trainers in ``nimbusml`` support additional types of data
sources as inputs, besides arrays and matrices.
The supported data sources are:

* ``list`` - for dense data
* ``numpy.ndarray`` and ``numpy.array`` - for dense data
* ``scipy.sparse_csr`` - for sparse data
* ``pandas.DataFrame`` and ``pandas.Series`` - for dense data with a schema
* ``FileDataStream`` - for dense data with a schema. 


Data in Lists
"""""""""""""

A list is a natural way to represent data layout.

Most trainers accept a list of values for X and y, as shown in the example below. The values
should be valid. If NaNs are present, they need to be imputed or filtered before feeding to a learner.
Additionally, the dimensions of X and y should be congruent, and the number of elements in the
examples should be identical and of the same type.

For transforms, the data dimension and composition requirements are less rigid, and depend
entirely on the transform.

Example:
::
    from nimbusml.linear_model import LogisticRegressionBinaryClassifier
    X = [[.1, .2],[.2, .3]]
    y = [0,1]
    LogisticRegressionBinaryClassifier().fit(X,y).predict(X)


Data in Numpy Arrays
""""""""""""""""""""

The data source can also be a ``numpy.array`` or ``numpy.ndarray`` object. The dimension and data
composition requirements are similar to the ones for lists.

Example:
::
    import numpy as np
    from nimbusml.linear_model import LogisticRegressionBinaryClassifier

    X = np.array([[.1, .2],[.2,.3]])
    y = np.array([0,1])
    LogisticRegressionBinaryClassifier().fit(X,y).predict(X)

Data in DataFrames and Series
"""""""""""""""""""""""""""""

Data in ``pandas.DataFrame`` and ``pandas.Series`` classes may also be used with trainers and
transforms. One advantage of dataframes is that column names can be user defined and used to
specify which columns should be transformed (see :ref:`l-pipeline-syntax`).

Example:
::

    import pandas as pd
    from nimbusml.linear_model import LogisticRegressionBinaryClassifier
    X = pd.DataFrame(data=dict(
            Sepal_Length=[2.5, 2.6],
            Sepal_Width=[.75, .9],
            Petal_Length=[2.5, 2.5],
            Petal_Width=[.8, .7]))

    y = pd.DataFrame([0,1])
    LogisticRegressionBinaryClassifier().fit(X,y).predict(X)


.. _datasources_file:

Data from a FileDataStream
""""""""""""""""""""""""""

Data in a file can be processed directly without preloading into memory. The data can be streamed efficiently using
:py:class:`nimbusml.FileDataStream` class, which replaces the X and y arguments of
the ``fit()`` and ``predict()`` methods of trainers. Users can create a
:py:class:`nimbusml.FileDataStream` class using the ``FileDataStream.read_csv()`` function or based on a ``DataSchema``.
More details about constructing a :py:class:`nimbusml.DataSchema` is discussed in :ref:`schema`.   


Example:
::
    from nimbusml.datasets import get_dataset
    from nimbusml import Pipeline, FileDataStream, DataSchema
    from nimbusml.ensemble import LightGbmClassifier

    path = get_dataset('infert').as_filepath()

    schema = DataSchema.read_schema(path, sep=',')
    ds = FileDataStream(path, schema = schema)
    
    #Equivalent to
    #ds = FileDataStream.read_csv(path, sep=',')

    pipeline = Pipeline([
        LightGbmClassifier(feature=['age', 'parity', 'induced'], label='case')
        ])

    pipeline.fit(ds)
    pipeline.predict(ds)

Output Data Types of Transforms
-------------------------------

When used inside a `sklearn.pipeline.Pipeline
<https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html>`_,
the return type of all of the transforms is a ``pandas.DataFrame``.

When used individually or inside a :py:class:`nimbusml.Pipeline`
that contains only transforms, the default output is a ``pandas.DataFrame``. To instead output an
`IDataView <https://github.com/dotnet/machinelearning/blob/master/docs/code/IDataViewImplementation.md>`_,
pass ``as_binary_data_stream=True`` to either ``transform()`` or ``fit_transform()``.
To output a sparse CSR matrix, pass ``as_csr=True``.
See :py:class:`nimbusml.Pipeline` for more information.

Note, when used inside a :py:class:`nimbusml.Pipeline`, the outputs are often stored in
a more optimized :ref:`VectorDataViewType`, which minimizes data conversion to
dataframes. When several transforms are combined inside an :py:class:`nimbusml.Pipeline`,
the intermediate transforms will store the data in the optimized format and only
the last transform will return a ``pandas.DataFrame`` (or IDataView/CSR; see above).


