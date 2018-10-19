# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
"""
General export functions.
"""


def dot_export_pipeline(pipeline, X, y=None, **params):
    """
    Exports a pipeline in `DOT language
    <https://en.wikipedia.org/wiki/DOT_(graph_description_language)>`_.
    Relies on method :py:meth:`get_fit_info
    <nimbusml.pipeline.Pipeline.get_fit_info>`.

    The function shows intermediate columns between operators.
    Blue columns are left unchanged, yellow columns are either
    created or replaced.

    ::

        import pandas
        from nimbusml.linear_model import FastLinearRegressor
        from nimbusml.feature_extraction.categorical import OneHotVectorizer
        from nimbusml.preprocessing.normalization import MeanVarianceScaler
        from nimbusml.preprocessing.schema import ColumnDropper
        from nimbusml import Pipeline
        from nimbusml.utils.exports import dot_export_pipeline

        df = pandas.DataFrame(dict(education=['A', 'B', 'A', 'B', 'A'],
                                   workclass=['X', 'X', 'Y', 'Y', 'Y'],
                                   yy=[1.1, 2.2, 1.24, 3.4, 3.4]))
        X = df.drop('yy', axis=1)
        y = df['yy']

        exp = Pipeline([
                    MeanVarianceScaler() << {'new_y': 'yy'},
                    OneHotVectorizer() << ['workclass', 'education'],
                    ColumnDropper() << 'yy',
                    FastLinearRegressor() << {'Feature': ['workclass',
                                                          'education'],
                                               Role.Label: 'new_y'}
                    ])

        dot = dot_export_pipeline(exp, X, y)
        print(doc)

    :func:`img_export_pipeline` uses this function
    to render the graph as an image.

    .. image:: pipe.gv.png
        :width: 400
    """
    exp = ["digraph{", "  orientation=portrait;"]
    info = pipeline.get_fit_info(X, y=y, **params)[0]
    columns = {}
    fontsize = 8

    for i, line in enumerate(info):
        if i == 0:
            schema = line['schema_after']
            labs = []
            for c, col in enumerate(schema):
                columns[col] = 'sch0:f{0}'.format(c)
                labs.append("<f{0}> {1}".format(c, col))
            node = '  sch0[label="{0}",shape=record,fontsize={1}];'.format(
                "|".join(labs), params.get('fontsize', fontsize))
            exp.append(node)

        else:
            if line['name'] == 'ColumnDropper':
                # We skip as it does not introduce any data.
                continue

            exp.append('')
            if line['type'] == 'transform':
                node = '  node{0}[label="{1}",shape=box,style="filled' \
                    ',rounded",color=cyan,fontsize={2}];'.format(
                        i, line['name'],
                        int(params.get('fontsize', fontsize) * 1.5))
            else:
                node = '  node{0}[label="{1}",shape=box,style="filled,' \
                        'rounded",color=yellow,fontsize={2}];'.format(
                            i, line['name'],
                            int(params.get('fontsize', fontsize) * 1.5))
            exp.append(node)

            for inps in line['inputs']:
                if ':' in inps:
                    spl = inps.split(':')
                    role = spl[0]
                    inpl = spl[-1].split(',')
                else:
                    inpl = [inps]
                    role = None

                for inp in inpl:
                    nc = columns[inp]
                    if role is not None:
                        edge = '  {0} -> node{1} [label="{2}"' \
                            ',fontsize={3}];'.format(
                                nc, i, role, fontsize)
                    else:
                        edge = '  {0} -> node{1};'.format(nc, i)
                    exp.append(edge)

            labs = []
            for c, out in enumerate(line['outputs']):
                columns[out] = 'sch{0}:f{1}'.format(i, c)
                labs.append("<f{0}> {1}".format(c, out))
            node = '  sch{0}[label="{1}",shape=record,fontsize={2}];'.format(
                i, "|".join(labs), params.get('fontsize', fontsize))
            exp.append(node)

            for out in line['outputs']:
                nc = columns[out]
                edge = '  node{1} -> {0};'.format(nc, i)
                exp.append(edge)

    exp.append('}')
    return "\n".join(exp)


def img_export_pipeline(pipeline, X, y=None, **params):
    """
    Produces an image which represents the data
    and the pipelines steps. It converts the export
    returned by function :func:`dot_export_pipeline`
    and returns a graph built by module
    `graphviz <https://graphviz.readthedocs.io/>`_.

    ::

        import pandas
        from nimbusml.linear_model import FastLinearRegressor
        from nimbusml.feature_extraction.categorical import OneHotVectorizer
        from nimbusml.preprocessing.normalization import MeanVarianceScaler
        from nimbusml.preprocessing.schema import ColumnDropper
        from nimbusml import Pipeline
        from nimbusml.utils.exports import img_export_pipeline

        df = pandas.DataFrame(dict(education=['A', 'B', 'A', 'B', 'A'],
                                   workclass=['X', 'X', 'Y', 'Y', 'Y'],
                                   yy=[1.1, 2.2, 1.24, 3.4, 3.4]))
        X = df.drop('yy', axis=1)
        y = df['yy']

        exp = Pipeline([
                    MeanVarianceScaler() << {'new_y': 'yy'},
                    OneHotVectorizer() << ['workclass', 'education'],
                    ColumnDropper() << 'yy',
                    FastLinearRegressor() << {'Feature': ['workclass',
                                                          'education'],
                                              Role.Label: 'new_y'}
                    ])

        img_export_pipeline(exp, X, y).render("mypipeline.png")

    .. image:: pipe.gv.png
        :width: 400
    """
    from graphviz import Source
    dot = dot_export_pipeline(pipeline, X, y=y, **params)
    gr = Source(dot)
    return gr
