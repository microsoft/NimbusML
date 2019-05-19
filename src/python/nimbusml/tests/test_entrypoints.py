"""
test low-level entrypoints
"""
import unittest

from nimbusml.internal.entrypoints \
    .trainers_logisticregressionbinaryclassifier import \
    trainers_logisticregressionbinaryclassifier
from nimbusml.internal.entrypoints.transforms_featurecombiner import \
    transforms_featurecombiner
from nimbusml.internal.entrypoints.transforms_twoheterogeneousmodelcombiner \
    import \
    transforms_twoheterogeneousmodelcombiner
from nimbusml.internal.utils.entrypoints import EntryPoint, Graph


# from imp import reload
# reload(microsoftml.entrypoints.trainers_logisticregressionbinaryclassifier)


class TestEntryPoints(unittest.TestCase):

    def test_transforms_featurecombiner(self):
        # import pdb; pdb.set_trace()
        # args
        data = "$input_data"
        features = ["feature1", "feature2"]
        output_data = "$training_data"
        model = "$transform_model"
        # call
        node = transforms_featurecombiner(
            data=data, features=features, output_data=output_data, model=model
        )
        # check
        assert isinstance(node, EntryPoint)
        assert node.inputs["Data"] == data
        assert node.inputs["Features"] == features
        assert node.outputs["OutputData"] == output_data
        assert node.outputs["Model"] == model
        assert node.input_variables == {data}
        assert node.output_variables == {output_data, model}

    def test_trainers_logisticregressionbinaryclassifier(self):
        # import pdb; pdb.set_trace()
        # args
        training_data = "$training_data"
        quiet = False
        label_column = "labelColumn"
        predictor_model = "$predictor_model"
        # call
        node = trainers_logisticregressionbinaryclassifier(
            training_data=training_data,
            quiet=quiet,
            label_column_name=label_column,
            predictor_model=predictor_model)
        # check
        assert isinstance(node, EntryPoint)
        assert node.inputs["TrainingData"] == training_data
        assert node.inputs["Quiet"] == quiet
        assert node.inputs["LabelColumnName"] == label_column
        assert node.input_variables == {training_data}
        assert node.output_variables == {predictor_model}

    def test_transforms_twoheterogeneousmodelcombiner(self):
        # import pdb; pdb.set_trace()
        # args
        transform_model = "$transform_model"
        predictor_model = "$predictor_model"
        model = "$output_model"
        # call
        node = transforms_twoheterogeneousmodelcombiner(
            transform_model=transform_model, predictor_model=predictor_model,
            model=model)
        # check
        assert isinstance(node, EntryPoint)
        assert node.inputs["TransformModel"] == transform_model
        assert node.inputs["PredictorModel"] == predictor_model
        assert node.outputs["PredictorModel"] == model
        assert node.input_variables == {transform_model, predictor_model}
        assert node.output_variables == {model}

    def test_logistic_regression_graph(self):
        # import pdb; pdb.set_trace()
        # args
        data = "$input_data"
        features = ["xint1"]
        output_data = "$training_data"
        model = "$transform_model"
        # call
        feature_node = transforms_featurecombiner(
            data=data, features=features, output_data=output_data, model=model
        )
        # args
        training_data = "$training_data"
        quiet = False
        label_column = "ylogical"
        predictor_model = "$predictor_model"
        # call
        lr_node = trainers_logisticregressionbinaryclassifier(
            # , FeatureColumn = "Features"
            training_data=training_data, quiet=quiet,
            label_column=label_column, predictor_model=predictor_model
        )
        # args
        transform_model = "$transform_model"
        predictor_model = "$predictor_model"
        model = "$output_model"
        # call
        combine_node = transforms_twoheterogeneousmodelcombiner(
            transform_model=transform_model, predictor_model=predictor_model,
            model=model)
        # compose graph
        # graph_sub = Graph(feature_node, lr_node, combine_node)
        # print(graph_sub)
        all_nodes = [feature_node, lr_node, combine_node]
        graph = Graph(
            dict(
                input_data=""), dict(
                output_model=""), False, *all_nodes)
        # print(graph)
        graph.run(X=None, dryrun=True)

        # lr = graph.run(formula = "ylogical ~ xint1", data = ds
        #    , blocks_per_read = 1, report_progress = True
        #    )
