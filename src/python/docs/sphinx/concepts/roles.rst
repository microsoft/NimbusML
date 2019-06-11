.. rxtitle:: Column Roles for Trainers
.. rxdescription:: How to select columns for operation of learners

.. _roles:

=========================
Column Roles for Trainers
=========================

.. contents::
    :local:


Roles and Learners
------------------

Columns play different roles in the context of trainers. NimbusML supports the following roles, as defined in :py:class:`nimbusml.Role`

* Role.Label - the column representing the dependent variable.
* Role.Feature - the column(s) representing the independent variable(s).
* Role.Weight - the weights column.
* Role.GroupId - the column containing grouping values for ranking.

The ``<<`` operator is used to tell the trainer which columns should play which role. When roles
are assigned to the trainer in a pipeline, they take precendence over the position of arguments in
the ``fit()`` method of the pipeline. Typically  ``fit(X, y)`` denotes that X are the features and y
are the labels. However, if roles are set for a trainer, then you can simply invoke ``fit(X)``, and
the trainer will use the columns in X as per the defined roles. Note that X can be any valid
:ref:`datasources`, including :py:class:`nimbusml.FileDataStream`, as long as the
columns can be referenced by name.

The trainer will exclude columns with roles Role.Label, Role.Weight, Role.GroupId (if any are
specified), and use all remaining columns of the input data as features. If Role.Feature is specified, only those
columns will be used as features, and the remaining columns will be ignored.

Example of Label and Feature Role
"""""""""""""""""""""""""""""""""

Roles are especially useful when the modeling data needs to be generated dynamically. The example
below creates a column *new_y* and assigns it as the target variable, using normalized values of the
orginal *y*.

::

    from nimbusml import Role
    from nimbusml import Pipeline
    from nimbusml.feature_extraction.categorical import OneHotVectorizer
    from nimbusml.linear_model import FastLinearRegressor
    from nimbusml.preprocessing.normalization import MeanVarianceScaler
    import pandas

    df = pandas.DataFrame(dict(education=['A', 'B', 'A', 'B', 'A'],
                               workclass=['X', 'X', 'Y', 'Y', 'Y'],
                               y=[1.1, 2.2, 1.24, 3.4, 3.4]))

    pipe = Pipeline([
        MeanVarianceScaler() << {'new_y': 'y'},
        OneHotVectorizer() << ['workclass', 'education'],
        FastLinearRegressor() << {Role.Label:'new_y', Role.Feature:['workclass', 'education']}
        #Equivalent to << {'Label':'new_y', 'Feature':['workclass', 'education']}, no need to import Role class
    ])
    pipe.fit(df)

    scores = pipe.predict(df)


The roles can be also be set using arguments to the trainer explicitly, instead of using the
``<<`` operator, as in the example below.

::

    df = pandas.DataFrame(dict(education=['A', 'B', 'A', 'B', 'A'],
                               workclass=['X', 'X', 'Y', 'Y', 'Y'],
                               y=[1.1, 2.2, 1.24, 3.4, 3.4]))

    pipe = Pipeline([
        MeanVarianceScaler(columns={'new_y': 'y'}), # renaming output column
        OneHotVectorizer(columns=['workclass', 'education']), # keep the same name
        FastLinearRegressor(label='new_y', feature=['workclass', 'education'])
    ])
    pipe.fit(df)

    scores = pipe.predict(df)



Example of Weight Role
"""""""""""""""""""""""

Most of the learners can make use of observation weights. This allows each instance in the dataset
to be assigned an individual weight. The weight is a non-negative real number indicating the relative 
importance of this instance over the others. The following example illustrates how to use weights 
without using the ``<<`` operator.

::

    df = pandas.DataFrame(dict(education=['A', 'B', 'A', 'B', 'A'],
                               workclass=['X', 'X', 'Y', 'Y', 'Y'],
                               weights=[1., 1., 1., 2., 1.],
                               y=[1.1, 2.2, 1.24, 3.4, 3.4]))
    
    exp = Pipeline([
        MeanVarianceScaler(columns={'new_y': 'y'}),
        OneHotVectorizer(columns=['workclass', 'education']),
        FastTreesRegressor(feature=['workclass', 'education'], label='new_y', weight='weights')
    ])
    exp.fit(df)
    prediction = exp.predict(df)

It can indicated to the learner by assigning the column a role using the ``<<`` operator as follows.

::

    exp = Pipeline([
        MeanVarianceScaler() << {'new_y': 'y'},
        OneHotVectorizer() << ['workclass', 'education'],
        FastTreesRegressor() << {Role.Feature:['workclass', 'education'], Role.Label: 'new_y', Role.Weight: 'weights'}
        #Equivalent to << {'Feature':['workclass', 'education'], 'Label': 'new_y', 'Weight': 'weights'}
    ])
    exp.fit(df)
    prediction = exp.predict(df)


Example of GroupId Role
"""""""""""""""""""""""

Same goes for the group. Rankers needs the GroupId to link rows to rank. A ranker for search engine needs a
dataset with a row per displayed result. The GroupId is ued to tell the learner which results belong to the
same query, to group together the candidate set of documents for a single query. NimbusML needs features,
a target (relevance label of the result) and a GroupId.

Below is an example of using GroupId at the trainer.

::

    df = pandas.DataFrame(dict(education=['A', 'B', 'A', 'B', 'A'],
                               workclass=['X', 'X', 'Y', 'Y', 'Y'],
                               group=[1, 1, 2, 2, 2],
                               y=[1.1, 2.2, 1.24, 3.4, 3.4]))
    
    exp = Pipeline([
        OneHotVectorizer() << ['workclass', 'education'],
        ToKey() << 'group',
        LightGbmRanker(minimum_example_count_per_leaf = 1)   << {Role.Feature: ['workclass', 'education'], Role.Label:'y', Role.GroupId:'group'}
        #Equivalent to LightGbmRanker(minimum_example_count_per_leaf = 1)   << {'Feature': ['workclass', 'education'], 'Label':'y', 'GroupId':'group'}
        #Equivalent to LightGbmRanker(minimum_example_count_per_leaf = 1, feature = ['workclass', 'education'], label = 'y', group_id = 'group')
    ])
    exp.fit(df)
    prediction = exp.predict(df)