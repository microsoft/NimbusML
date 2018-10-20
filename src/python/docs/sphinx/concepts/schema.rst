.. rxtitle:: Schema
.. rxdescription:: Schema to describe input FileDataStream

.. _schema:

======
Schema
======

.. index:: column type

.. contents::
    :local:
    :depth: 1

Introduction to Schema
----------------------

The NimbusML data framework relies on a schema to understand the column names and mix of column
types in the dataset, which may originate from any of the supported :ref:`datasources`. It is 
automatically inferred when a :py:class:`nimbusml.FileDataStream` or :py:class:`nimbusml.DataSchema` is created.

Transforms have the ability to operate on subsets of columns in the dataset, as well as alter the
resulting output schema, which effects other transforms downstream. For users, it would be very useful to 
understand how NimbusML processes the data in a pipeline for debugging purposes or training the model with :py:class:`nimbusml.FileDataStream`.

The schema comes with two formats for its representation, (1) object representation and (2) string format. After generating a :py:class:`nimbusml.FileDataStream`, users can view the 
object representation of the schema by using ``repr()`` function:

::

            from nimbusml import FileDataStream
            import numpy as np
            import pandas as pd
            
            data = pd.DataFrame(dict(real = [0.1, 2.2], text = ['word','class'], y = [1,3]))
            data.to_csv('data.csv', index = False, header = True)
            
            ds = FileDataStream.read_csv('data.csv', collapse = True,
                                            numeric_dtype = np.float32, sep = ',')
            print(repr(ds.schema))
            #DataSchema([DataColumn(name='real', type='R4', pos=0),
            #            DataColumn(name='text', type='TX', pos=1), 
            #            DataColumn(name='y', type='R4', pos=2)], 
            #            header=True, 
            #            sep=',')

The name, type, position of the columns are shown as well as the information about if the data has a header or what
the seperation of the columns is. It is always useful for users to examine the Schema of a :py:class:`nimbusml.FileDataStream` before training
the model. 

As can be seen in the above example, the arguments for the :py:func:`nimbusml.FileDataStream.read_csv` are used
to modify the Schema of the generated :py:class:`nimbusml.FileDataStream`. More details about how
to modify the Schema is presented in [DataSchema Alterations](schema.md#dataschema-alterations). All the arguments
discussed in this section are also applicable for :py:class:`nimbusml.FileDataStream`.

The schema string format looks like a series of entries shown below: 

::

   col=<name>:<type>:<position> [options]

where

* **col=** is specified for every column in the dataset,
* **name** is the name of the column,
* **position** is the 0-based index (or index range) of the column(s),
* **type** is one of the :ref:`column-types`. When the *position* is a range (i.e. *start_index-end_index*), the column is of :ref:`VectorType`.
* **options**

  * **header=** [+-] : Specifies if there is a header present in the text file

  * **sep=** [delimiter] : the delimiter for the columns

For instance,
::

   schema = 'sep=, col=Features:R4:0-2 col=Label:R4:3 col=Text:TX:4 header+'

   schema = 'sep=tab col=Sentiment:BL:1 col=SentimentSource:TX:2 col=SentimentText:TX:3 col=rownum:R4:4 header=+'

   schema = 'sep=, col=Features:R4:0-4 col=UniqueCarrier:TX:5 col=Origin:TX:6 col=Dest:TX:7 col=Label:BL:9 header=+'

The first example indicates that the data is seperated by ``,``, the first three columns (with index ranging from 0 to 2) are named *Features* and with type **R4**, i.e. single precision floating-point.
The fourth column (with index 3) is named *Label* and with type **R4**. The fifth column (with index 4) is named *Text* and with type **TX**. The data has a header.


DataSchema Class
----------------

The :py:class:`nimbusml.DataSchema` class can be used to automatically infer the schema from the different data sources.

Example of Schema for List
""""""""""""""""""""""""""

Lists are the simplest source of data. The schema inferred below shows that the values are
treated as a single column with name *Unknown* of type TX, starting at index 0. The header=+
indicates that there is a header row in the data.


::


    import numpy as np
    from pandas import DataFrame
    from nimbusml import DataSchema

    list = [[1.0, 1.0, 2.0], [3.0, 5.0, 6.0]]
    schema = DataSchema.read_schema(list)
    print(repr(schema))
    #DataSchema([DataColumn(name='c0', type='R8', pos=0),
    #            DataColumn(name='c1', type='R8', pos=1), 
    #            DataColumn(name='c2',type='R8', pos=2)], 
    #            header=True)
    print(schema)
    #col=c0:R8:0 col=c1:R8:1 col=c2:R8:2 header=+


Example of Schema for numpy.array
"""""""""""""""""""""""""""""""""

The DataSchema class infers that there is a header row in the dataset, and there are 3 columns,
all of type R4 with index range of 0 to 2. When the type is changed from float32 to int16, the
schema changes accoringly.

::

    arr = np.array(list).astype(np.float32)
    schema = DataSchema.read_schema(arr)
    print(repr(schema))
    #DataSchema([DataColumn(name='Data', type='R4', pos=(0, 1, 2))],
    #            header=True)
    print(schema)
    #col=Data:R4:0-2 header=+


    arr = np.array(list).astype(np.int16)
    schema = DataSchema.read_schema(arr)
    print(repr(schema))
    #DataSchema([DataColumn(name='Data', type='I2', pos=(0, 1, 2))],
    #            header=True)
    print(schema)
    #col=Data:I2:0-2 header=+


Example of Schema for pandas.DataFrame
""""""""""""""""""""""""""""""""""""""

The DataSchema class infers that there is a header row in the dataset, and there are 3 columns,
all of types R8, I8 and TX, with column names *X1*, *X2* and *X3*.

::


    df = DataFrame(dict(X1=[0.1, 0.2], X2=[1, 2], X3=["a", "b"]))
    schema = DataSchema.read_schema(df)
    print(repr(schema))
    #DataSchema([DataColumn(name='X1', type='R8', pos=0),
    #            DataColumn(name='X2', type='I8', pos=1), 
    #            DataColumn(name='X3',type='TX', pos=2)], 
    #            header=True)
    print (schema)
    #col=X1:R8:0 col=X2:I8:1 col=X3:TX:2 header=+


.. _schema_moredetails:

Example of Schema for a File
""""""""""""""""""""""""""""""""""""""

The transforms and trainers in NimbusML support various :ref:`datasources` as inputs.
When the data is in a ``pandas.DataFrame``, the schema is inferred automatically from the
``dtype`` of the columns.

When the data is in a file, the schema will be inferred when creating a :py:class:`nimbusml.FileDataStream` using ``read_csv()`` or 
using ``nimbusml.DataSchema.read_schema()``. [Update when methods are included in API].

Example (from file):
::

    from nimbusml import DataSchema
    from pandas import DataFrame
    from collections import OrderedDict
    data = DataFrame(OrderedDict(real1=[0.1, 0.2], real2=[0.1, 0.2], integer=[1, 2], text=["a", "b"]))
    # write dataframe to file
    data.to_csv('data.txt', index=False)

    # infer schema directly from file
    schema = DataSchema.read_schema('data.txt')
    print(repr(schema))
    #DataSchema([DataColumn(name='real1', type='R8', pos=0),
    #            DataColumn(name='real2', type='R8', pos=1),
    #            DataColumn(name='integer', type='I8', pos=2),
    #            DataColumn(name='text', type='TX', pos=3)], header=True)
    print(schema)
    #col=real1:R8:0 col=real2:R8:1 col=integer:I8:2 col=text:TX:3 header=+

DataSchema Alterations
----------------------

Merge Consecutive Columns of Same Type
""""""""""""""""""""""""""""""""""""""

Data may consist of numerous columns of the same type, and often it's convenient to group them
under a single name. The :py:class:`nimbusml.DataSchema` provides the ``collapse``
argument to shorten the schema representation by grouping homongenous types.

Example:
::

    schema = DataSchema.read_schema('data.txt', collapse=True)
    print(repr(schema))
    #DataSchema([DataColumn(name='real1', type='R8', pos=(0, 1)),
    #            DataColumn(name='integer', type='I8', pos=2),
    #            DataColumn(name='text', type='TX', pos=3)], header=True)
    print(schema)
    #col=real1:R8:0-1 col=integer:I8:2 col=text:TX:3 header=+

We see that columns *real* and *real2* are merged into a single one ``col=real1:R8:0-1``. It is not a
real anymore but a vector of two floats. Every learner uses features encoded as a vector of
features. Every transform in a pipeline would convert text, categories, floats into feature vectors. It is faster to do that
at loading time. The parameter ``collapse=True`` forces the function to merge consecutive columns
with the same type into vectors.


Merge All Columns of Same Type
""""""""""""""""""""""""""""""

If ``collapse == 'all'``, it merges all columns of the same type unless specified in argument ``names``. Let's see an example:

.. rxexample::
    :execute:
    :print_output:

    from nimbusml.datasets import get_dataset
    from pandas import read_csv
    path = get_dataset("infert").as_filepath()
    df = read_csv(path)
    print(df.head(n=2))

*case* is the target, eveything else must be features if numeric. We want to merge every column into
*Features* except *row_num* (row index), *education* (text) and *case* (target). *education* is not
merged by default as it is not a numerical column.

Example:
::

    import numpy as np
    schema = DataSchema.read_schema(path, collapse='all', sep=',',
                                         numeric_dtype=np.float32, #convert all numeric columns to R4
                                         names={0:'row_num', 5:'case'})
    print(repr(schema))
    #DataSchema([DataColumn(name='row_num', type='R4', pos=0),
    #            DataColumn(name='education', type='TX', pos=1),
    #            DataColumn(name='age', type='R4', pos=(2, 3, 4, 6, 7, 8)),
    #            DataColumn(name='case', type='R4', pos=5)], header=True, sep=',')
    print(schema)
    #col=row_num:R4:0 col=education:TX:1 col=age:R4:2-4,6-8 col=case:R4:5 header=+ sep=,


Changing a column name
""""""""""""""""""""""

Some datasets have many columns and it is convenient to modify the first ones and let the function
handle the rest. Below is an example of how to modify column names.

Example:
::

    schema = DataSchema.read_schema('data.txt', collapse=True, sep=',',
                           names={0: 'newname', 1: 'newname2'})
    print(repr(schema))
    #DataSchema([DataColumn(name='newname', type='R8', pos=0),
    #            DataColumn(name='newname2', type='R8', pos=1),
    #            DataColumn(name='integer', type='I8', pos=2),
    #            DataColumn(name='text', type='TX', pos=3)], header=True, sep=',')
    print(schema)
    #col=newname:R8:0 col=newname2:R8:1 col=integer:I8:2 col=text:TX:3 header=+

Next example renames from column 0 to column 1 into *real_0*, *real_1*, ...

Example:
::

    schema = DataSchema.read_schema('data.txt', collapse=False, sep=',',
                           names={(0,1): 'real'})
    print(repr(schema))
    #DataSchema([DataColumn(name='real_0', type='R8', pos=0),
    #            DataColumn(name='real_1', type='R8', pos=1),
    #            DataColumn(name='integer', type='I8', pos=2),
    #            DataColumn(name='text', type='TX', pos=3)], header=True, sep=',')
    print(schema)
    #col=real_0:R8:0 col=real_1:R8:1 col=integer:I8:2 col=text:TX:3 header=+

Changing a column type
""""""""""""""""""""""

The ``read_schema()`` method uses the ``dtype`` argument to change all types or only a few.
We can also use ``numeric_dtype=np.float32`` to change all numeric columns to R4 type.

Example:
::

    schema = DataSchema.read_schema('data.txt', collapse=True, sep=',',
                                    dtype={'real1': np.float32})
    print(repr(schema))
    #DataSchema([DataColumn(name='real1', type='R4', pos=0),
    #            DataColumn(name='real2', type='R8', pos=1),
    #            DataColumn(name='integer', type='I8', pos=2),
    #            DataColumn(name='text', type='TX', pos=3)], header=True, sep=',')
    print(schema)
    #col=real1:R4:0 col=real2:R8:1 col=integer:I8:2 col=text:TX:3 header=+

Other Arguments
""""""""""""""""""""""

The ``sep`` argument can be used to specify another separator besides ``','``, which is the default
delimiter. The user can also manually play with the schema himself.

Example:
::

    for col in schema:
        print(type(col), col)
    #<class 'nimbusml.internal.utils.data_schema.DataColumn'> col=real1:R4:0
    #<class 'nimbusml.internal.utils.data_schema.DataColumn'> col=real2:R8:1
    #<class 'nimbusml.internal.utils.data_schema.DataColumn'> col=integer:I8:2
    #<class 'nimbusml.internal.utils.data_schema.DataColumn'> col=text:TX:3

One Complex Example
"""""""""""""""""""

In this section, we only show the string representation of the schema for simplicity.
Ranking models require three kind of columns. Two of the columns are the typical *Features* and
*Label* columns (of numeric type **R4** == ``numpy.float32``) and a third *GroupId* column which ties
all observations to a specific ranking group. Note that all examples with the same *GroupId* must
appear sequentially and its type must be **TX** == ``str``. When reading the file without any additional
information, the raw schema is the following::

    col=c0:I8:0 col=c1:I8:1 col=c2:I8:2 col=c3:I8:3 col=c4:I8:4 ... header=- sep=,

But we need to have this::

    col=Label:R4:0 col=GroupId:TX:1 col=Features:R4:2-2109 header=- sep=,

Let's see step by step how to get that and it starts with the raw schema generated using ``read_schema()``:

Example:
::

    from nimbusml import DataSchema
    from nimbusml.datasets import get_dataset

    path = get_dataset('gen_tickettrain').as_filepath()
    schema = DataSchema.read_schema(path, sep=',')
    print(str(schema))
    #col=rank:I8:0 col=group:I8:1 col=carrier:TX:2 col=price:I8:3 col=Class:I8:4 
    #col=dep_day:I8:5 col=nbr_stops:I8:6 col=duration:R8:7 header=+ sep=,
    
Let's rename label and group id:

Example:
::

    schema = DataSchema.read_schema(path, sep=',', header=True,
                        names={0:'Label', 1:'GroupId'})    # added
    print(str(schema))
    #col=Label:I8:0 col=GroupId:I8:1 col=carrier:TX:2...

Let's change the column types. However, this requires to change the type of more than 2000 columns.
As types can be changed given a column name and not its position, we use a regular expression to do
so.

Example:
::

    schema = DataSchema.read_schema(path, sep=',',
                        names={0:'Label', 1:'GroupId'},
                        dtype={'GroupId': str, 'Label': np.float32})    # added
    print(str(schema))
    #col=Label:R4:0 col=GroupId:TX:1 col=carrier:TX:2 col=price:I8:3


Let's then merge every columns used later as features into
a single name.

Example:
::

    schema = DataSchema.read_schema(path, sep=',', 
                        names={0:'Label', 1:'GroupId'},
                        dtype={'GroupId': str, 'Label': np.float32},
                        collapse = 'all')    # added
    print(str(schema))
    #col=Label:R4:0 col=GroupId:TX:1 col=carrier:TX:2 col=price:I8:3-6 col=duration:R8:7 header=+ sep=,


And finally, let's rename *c2* into *Features*:

Example:
::

    schema.rename('price', 'Features')    # added
    print(schema)
    #col=Label:R4:0 col=GroupId:TX:1 col=carrier:TX:2 col=Features:I8:3-6 col=duration:R8:7 header=+ sep=, #Voila!

Most of datasets are stored in text files. It is usually more convenient to load them in memory with
``pandas``. But when the datasets is too big, ``nimbusml`` has to directly load the data from its
location. It is more efficient to tell the parser which names and types it should use than changing
them by adding transforms in the pipeline. Given the :py:class:`nimbusml.DataSchema` generated 
above, a :py:class:`nimbusml.FileDataStream` can be created to train the model:

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