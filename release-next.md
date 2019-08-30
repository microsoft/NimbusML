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

## **Breaking Changes**

None.

## **Enhancements**

None.

## **Documentation and Samples**

None. 

## **Remarks**

None.
