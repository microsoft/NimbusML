.. rxtitle:: Pipelines
.. rxdescription:: Sklearn pipeline and nimbusml pipeline comparison

.. _Pipeline:

nimbusml.Pipeline() versus sklearn.Pipeline()
=======================================================

.. contents::
    :local:

This sections highlights the differences between using a `sklearn.Pipeline <http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html>`_ 
and :py:class:`nimbusml.Pipeline` to compose a sequence of transformers and/or trainers.

 
sklearn.Pipeline
----------------

``nimbusml`` transforms and trainers are designed to be compatible with
`sklearn.Pipeline <http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html>`_. 
For fully optimized performance and added functionality, it is recommended to use
:py:class:`nimbusml.Pipeline`. See below for more details.

.. _Experiment:

nimbusml.Pipeline
---------------------------

There are several advantages of using :py:class:`nimbusml.Pipeline` for experimentation.
It is recommended to use this version whenever a pipeline of transformations and trainers is
required for training models.

Support for Data Files
""""""""""""""""""""""

One requirement when using sklearn's version of Pipeline is that all the data must fit into memory. For
files that are too large to fit into memory, there is no easy way to train estimators directly by
streaming the examples one at a time.

The :py:class:`nimbusml.Pipeline` module accepts inputs X and y similarly to
`sklearn.Pipeline <http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html>`_, but also
inputs of type :py:class:`nimbusml.FileDataStream`, which is an optimized streaming file
reader class. This is highly recommended for large datasets. See [Data Sources](datasources.md#data-from-a-filedatastream) for an
example of using Pipeline with FileDataStream to read data in files.

Select which Columns to Transform
"""""""""""""""""""""""""""""""""

When using `sklearn.Pipeline <http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html>`_
the data columns of X and y (of type``numpy.array`` or ``scipy.sparse_csr``)
are anonymous and cannot be referenced by name. Operations and transformations are
therefore performed on all columns of the data.

In some cases, we may need to perform operations on only a few columns within X. This can be
achieved using :py:class:`nimbusml.Pipeline` by specifying a schema and setting the
arguments of transforms appropriately. See :ref:`l-pipeline-syntax` for details on column
operations.

.. seealso::

    :py:class:`nimbusml.DataSchema`,
    :py:class:`nimbusml.FileDataStream`

Optimized Chaining of Trainers/Transforms
"""""""""""""""""""""""""""""""""""""""""

Using NimbusML, trainers and transforms within a :py:class:`nimbusml.Pipeline` will
generally result in better performance compared to using them in a
`sklearn.Pipeline <http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html>`_.
Data copying is minimized when processing is limited to within the C# libraries, and if all
components are in the same pipeline, data copies between C# and Python is reduced.


