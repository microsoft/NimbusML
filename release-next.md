# [NimbusML](https://docs.microsoft.com/en-us/nimbusml/overview) Next

## **New Features**

- **Add initial implementation of DatasetTransformer.**

    [PR#240](https://github.com/microsoft/NimbusML/pull/240)
    This transform allows a fitted transformer based model to be inserted
    in to another `Pipeline`.

    ```python
    Pipeline([
        DatasetTransformer(transform_model=transform_pipeline.model),
        OnlineGradientDescentRegressor(label='c2', feature=['c1'])
    ])
    ```

## **Bug Fixes**

- **Fixed `classes_` attribute when no `y` input specified **

    [PR#218](https://github.com/microsoft/NimbusML/pull/218)
    Fix a bug with the classes_ attribute when no y input is specified during fitting.
    This addresses [issue 216](https://github.com/microsoft/NimbusML/issues/216)

- **Fixed Add NumSharp.Core.dll **

    [PR#220](https://github.com/microsoft/NimbusML/pull/220)
    Fixed a bug that prevented running TensorFlowScorer.
    This addresses [issue 219](https://github.com/microsoft/NimbusML/issues/219)

- **Fixed Enable scoring of ML.NET models saved with new TransformerChain format **

    [PR#230](https://github.com/microsoft/NimbusML/pull/230)
    Fixed error loading a model that was saved with mlnet auto-train.
    This addresses [issue 201](https://github.com/microsoft/NimbusML/issues/201)

- **Fixed Pass python path to Dprep package **

    [PR#232](https://github.com/microsoft/NimbusML/pull/232)
    Enable passing python executable to dataprep package, so dataprep can execute python transformations

- **Fixed `Pipeline.transform()` in transform only `Pipeline` fails if y column is provided **

    [PR#232](https://github.com/microsoft/NimbusML/pull/232)
    Enable calling `.transform()` on a `Pipeline` containing only transforms when the y column is provided 

## **Breaking Changes**

None.

## **Enhancements**

None.

## **Documentation and Samples**

None. 

## **Remarks**

None.
