# [NimbusML](https://docs.microsoft.com/en-us/nimbusml/overview) 1.3.0

## **New Features**

- **Save/Restore model when pickling Pipeline**

   [PR#189](https://github.com/microsoft/NimbusML/pull/189) Save and restore
   the underlying model file when pickling a nimbusml Pipeline.

- **Feature Contributions**

  [PR#196](https://github.com/microsoft/NimbusML/pull/196) Added support for
  observation level feature contributions. Exposes an API
  `Pipeline.get_feature_contributions()` that provides scores for how much
  each feature influenced a particular prediction, thereby allowing users to
  inspect which features were most important in making the prediction.

- **Add `classes_` to Pipeline**

   [PR#200](https://github.com/microsoft/NimbusML/pull/200) Add a `classes_`
   attribute to a Pipeline and/or predictor instance when calling
   `Pipeline.predict_proba()`.

- **Automatically Convert Input Of Handler, Filter and Indicator**

   [PR#204](https://github.com/microsoft/NimbusML/pull/204) Update Handler,
   Filter, and Indicator to automatically convert the input columns to float
   before performing the transform.

- **Combine Models**

   [PR#208](https://github.com/microsoft/NimbusML/pull/208) Add support for
   combining models from transforms, predictors and pipelines in to one model.

- **Azureml-Dataprep integration**

   [PR#181](https://github.com/microsoft/NimbusML/pull/181) Added support for
   dataflow objects as a datasource for pipeline training/testing.
   
- **Linear SVM Binary Classifier**
  [PR#180](https://github.com/microsoft/NimbusML/pull/180) Added
  `LinearSvmBinaryClassifier` in `nimbusml.linear_model`.
   
- **Ensemble Training**

  [PR#207](https://github.com/microsoft/NimbusML/pull/207) Enabled training of
  Ensemble models by adding `nimbusml.ensemble.EnsembleRegressor` and 
  `nimbusml.ensemble.EnsembleClassifier`. Added components needed
  to create ensemble models as new modules in `nimbusml.ensemble`. These
  components are passed as arguments to the ensemble trainers.
  - Preprocessing components for training multiple models to ensemble in 
  `nimbusml.ensemble.subset_selector` and  `nimbusml.ensemble.feature_selector`.
  - Post training components to create the ensemble from the trained models in
  `nimbusml.ensemble.sub_model_selector` and `nimbusml.ensemble.output_combiner`.

## **Bug Fixes**

- **Fixed memory leak**

   The [PR#184](https://github.com/microsoft/NimbusML/pull/184) fixed potentially
   large memory leak when transforming pandas dataframe.

- **Remove Stored References To `X` and `y`**

   [PR#195](https://github.com/microsoft/NimbusML/pull/195) Remove the stored
   references to X and y in BasePredictor.

- **Fixed Explicit `evaltype`**

   The [issue](https://github.com/microsoft/NimbusML/issues/193) where passing
   in an explicit `evaltype` to `_predict` in a NimbusML pipeline causes errors
   has been fixed with this
   [commit](https://github.com/microsoft/NimbusML/commit/1f97c9ef55f5e257f989db5f375cca5c55880258).

## **Breaking Changes**

None.

## **Enhancements**

None.

## **Documentation and Samples**

[Feature Contributions Example](https://github.com/microsoft/NimbusML/blob/master/src/python/nimbusml/examples/PipelineWithFeatureContributions.py)

LinearSvmBinaryClassifier Examples:
- [FileDataStream example](https://github.com/microsoft/NimbusML/blob/master/src/python/nimbusml/examples/LinearSvmBinaryClassifier.py)
- [DataFrame example](https://github.com/microsoft/NimbusML/blob/master/src/python/nimbusml/examples/examples_from_dataframe/LinearSvmBinaryClassifier_df.py)
  

## **Remarks**

None.
