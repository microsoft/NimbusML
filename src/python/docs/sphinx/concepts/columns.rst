.. rxtitle:: Column Operations for Transforms
.. rxdescription:: Select columns for operation for transforms

.. _l-pipeline-syntax:

================================
Column Operations for Transforms
================================

.. contents::
    :local:
    
How To Select Columns to Transform
----------------------------------

``nimbusml`` is compatible with the ``scikit-learn`` convention for column processing in ``fit()``,
``transform()`` and ``fit_transform()`` methods of trainers and transforms. By default, all
columns are transformed equally.

``nimbusml`` additionally provides a syntax to transform only a subset of columns. This is a useful
feature for many transforms, especially when the dataset containts columns of mixed types. For
example, a dataset with both numeric features and free text features. Similarly for trainers, the
concept of :ref:`roles` provides a mechanism to select which columns to use as labels and features.


Transform All Columns
"""""""""""""""""""""

By default, the ``OneHotVectorizer`` transform will process all columns, which in our example
results in a the original column values being replaced by their one hot encodings. Note that the
output of ``OneHotVectorizer`` are :ref:`VectorType`, so the output
names below are the column names appended with the ``slot`` names, which in our example are data
driven and generated dynamically from the input data.


.. rxexample::
    :execute:
    :print_output:

    import pandas as pd
    from nimbusml.feature_extraction.categorical import OneHotVectorizer

    # data with columns education and workclass
    X =  pd.DataFrame(data=dict( edu = ['bs', 'ms', 'phd', 'bs'],
                                 wclass= ['food', 'finance','food', 'movie'] ))
    xf = OneHotVectorizer()
    print( xf.fit_transform(X))

.. _use-operator-to-select-columns:

Use << Operator To Select Columns
"""""""""""""""""""""""""""""""""""

What if we only want to encode one of the columns? We simply use the ``<<`` operator to tell the
transform to restrict operations to the columns of interest. The ``<<`` operatator is syntactic
sugar for setting the ``columns`` argument of the transform.

All transforms in ``nimbusml`` have an implicit ``columns`` parameter to tell which columns to process,
and optionally how to name the output columns, if any. Refer to the reference sections for each
transform to see what format is allowed for the ``columns`` argument.

.. rxexample::
    :execute:
    :print_output:

    import pandas as pd
    from nimbusml.feature_extraction.categorical import OneHotVectorizer

    # data with columns education and workclass
    X =  pd.DataFrame(data=dict( edu = ['bs', 'ms', 'phd', 'bs'],
                                 wclass= ['food', 'finance','food', 'movie'] ))

    # use the << operator to select only edu to encode
    xf = OneHotVectorizer() << ['edu']
    print(xf.fit_transform(X))

.. _and-columns-are-interchangeable:

<< and columns= are interchangeable
"""""""""""""""""""""""""""""""""""

Let's see an example of setting the ``columns`` argument explicity, to get the same results as
using the ``<<`` operator.

.. rxexample::
    :execute:
    :print_output:

    import pandas as pd
    from nimbusml.feature_extraction.categorical import OneHotVectorizer

    # data with columns education and workclass
    X =  pd.DataFrame(data=dict( edu = ['bs', 'ms', 'phd', 'bs'],
                                 wclass= ['food', 'finance','food', 'movie'] ))

    # use `columns=` to do the same thing as `<<`
    xf = OneHotVectorizer(columns=['edu'])
    print(xf.fit_transform(X))


Renaming Output Columns of Transforms
"""""""""""""""""""""""""""""""""""""

Transformations are done in place, and therefore values in the original column will be replaced with
the updated values. To retain the original input column values, we can specify an optional output
column, with a different name than the input column, to store the transformed values.

Some columns may not allow renaming the output columns, so always refer to the reference sections
for each transform to see what format is allowed for the ``columns`` argument.

In the example below, the original *edu* column values are preserved, while the encoded values are
stored in the new column *xyz*, with slot name *bs*, *ms* and *phd*.

.. rxexample::
    :execute:
    :print_output:

    import pandas as pd
    from nimbusml.feature_extraction.categorical import OneHotVectorizer

    # data with columns education and workclass
    X =  pd.DataFrame(data=dict( edu = ['bs', 'ms', 'phd', 'bs'],
                                 wclass= ['food', 'finance','food', 'movie'] ))

    # let's retain the edu column, and create a
    # new output column xyz for the encoded values
    xf = OneHotVectorizer(columns={'xyz':'edu'})
    print('\n', xf.fit_transform(X))

Column Names in a Pipeline
""""""""""""""""""""""""""

Within a :py:class:`nimbusml.Pipeline`, there can be many transforms, each one
modifying column values, creating new columns and potentially deleting columns. The output of
each transform affects the data values and schema for the next transform in the pipeline.

In the example below, the original column values of *edu* are no longer available because
they are replaced with the encoded values. However the original values of *wclass* are still
available, because the encoded values are store in *A*.


.. rxexample::
    :execute:
    :print_output:

    import pandas as pd
    from nimbusml import Pipeline
    from nimbusml.feature_extraction.categorical import OneHotVectorizer

    # data with columns education and workclass
    X =  pd.DataFrame(data=dict( edu = ['bs', 'ms', 'phd', 'bs'],
                                 wclass= ['food', 'finance','food', 'movie'] ))

    pipe = Pipeline([
        OneHotVectorizer() << ['edu'],
        OneHotVectorizer() << {'A':'wclass'}
    ])
    print(pipe.fit_transform(X))






