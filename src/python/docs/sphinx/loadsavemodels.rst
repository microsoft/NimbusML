.. rxtitle:: Loading and Saving Models
.. rxdescription:: How to load and save models with nimbusml pipeline

.. _loadsavemodels:

==================================
Loading, Saving and Serving Models
==================================

.. index:: load save model



Persisting Models
-----------------

Trainers, transforms and pipelines can be persisted in a couple of ways. Using Python's built-in
persistence model of `pickle <https://docs.python.org/2/library/pickle.html>`_, or else by using the
the ``load_model()`` and ``save_model()`` methods of :py:class:`nimbusml.Pipeline`.

Advantages of using pickle is that all attribute values of objects are preserved, and can be
inspected after deserialization. However, for models trained from external sources such as the ML.NET C#
application, pickle cannot be used, and the ``load_model()`` method needs to be used instead.
Similarly the ``save_model()`` method saves the model in a format that can be used by external
applications.


Using Pickle
""""""""""""

Below is an example using pickle.

.. rxexample::
    :execute:
    :print_output:

    import pickle
    from nimbusml import Pipeline, FileDataStream
    from nimbusml.linear_model import AveragedPerceptronBinaryClassifier
    from nimbusml.datasets import get_dataset
    
    data_file = get_dataset('infert').as_filepath()
    
    ds = FileDataStream.read_csv(data_file)
    ds.schema.rename('case', 'case2') # column name case is not allowed in C#
    # Train a model and score
    pipeline = Pipeline([AveragedPerceptronBinaryClassifier(
        feature=['age', 'parity', 'spontaneous'], label='case2')])
    
    metrics, scores = pipeline.fit(ds).test(ds, output_scores=True)
    print(metrics)
    
    # Load model from file and evaluate. Note that 'evaltype'
    # must be specified explicitly
    s = pickle.dumps(pipeline)
    pipe2 = pickle.loads(s)
    metrics2, scores2 = pipe2.test(ds, evaltype='binary', output_scores=True)
    print(metrics2)

Using load_model() and save_model()
"""""""""""""""""""""""""""""""""""

Below is an example of using load_model() and save_model(). The model can also originate from
external tools such as the ML.NET C# application or Maml.exe command line tool. When loading a
model this way, the argument of 'evaltype' must be specified explicitly.

.. rxexample::
    :execute:
    :print_output:

    from nimbusml import Pipeline, FileDataStream
    from nimbusml.linear_model import AveragedPerceptronBinaryClassifier
    from nimbusml.datasets import get_dataset

    data_file = get_dataset('infert').as_filepath()
    ds = FileDataStream.read_csv(data_file)
    ds.schema.rename('case', 'case2') # column name case is not allowed in C#

    # Train a model and score
    pipeline = Pipeline([AveragedPerceptronBinaryClassifier(
        feature=['age', 'parity', 'spontaneous'], label='case2')])
    
    metrics, scores = pipeline.fit(ds).test(ds, output_scores=True)
    pipeline.save_model("mymodeluci.zip")
    print(metrics)

    # Load model from file and evaluate. Note that 'evaltype'
    # must be specified explicitly
    pipeline2 = Pipeline()
    pipeline2.load_model("mymodeluci.zip")
    metrics2, scores2 = pipeline2.test(ds, y = 'case2', evaltype='binary')
    print(metrics2)

Scoring in ML.NET
"""""""""""""""""""""""""""""""""""

The saved model ('mymodeluci.zip') can be used for scoring in ML.NET using the following code:

load_save_model_csharp to be inserted.