# [NimbusML](https://docs.microsoft.com/en-us/nimbusml/overview) Next

## **New Features**

- **Save/Restore model when pickling Pipeline**

   [PR#189](https://github.com/microsoft/NimbusML/pull/189) Save and restore
   the underlying model file when pickling a nimbusml Pipeline.

- **Feature Contributions**

  [PR#196](https://github.com/microsoft/NimbusML/pull/196) added support for observation level feature contributions. Exposes an API that provides scores for how much each feature influenced a particular prediction, thereby allowing users to inspect which features were most important in making the prediction.

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

## **Bug Fixes**

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
  

## **Remarks**

None.
