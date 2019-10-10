# [NimbusML](https://docs.microsoft.com/en-us/nimbusml/overview) Next

## **New Features**

- **Initial implementation of `csr_matrix` output support.**

    [PR#250](https://github.com/microsoft/NimbusML/pull/250)
    Add support for data output in `scipy.sparse.csr_matrix` format.

    ```python
    xf = OneHotVectorizer(columns={'c0':'c0', 'c1':'c1'})
    xf.fit(train_df)
    result = xf.transform(train_df, as_csr=True)
    ```
    
- **Permutation Feature Importance for model interpretibility.**

    [PR#279](https://github.com/microsoft/NimbusML/pull/279)
    Adds `permutation_feature_importance()` method to `Pipeline` and
    predictor estimators, enabling evaluation of model-wide feature
    importances on any dataset with same schema as the dataset used
    to fit the `Pipeline`.

    ```python
    pipe = Pipeline([
        LogisticRegressionBinaryClassifier(label='label', feature=['feature'])
    ])
    pipe.fit(data)
    pipe.permutation_feature_importance(data)
    ```

- **Initial implementation of DateTime input and output column support.**

    [PR#290](https://github.com/microsoft/NimbusML/pull/290)
    Add initial support for input and output of Pandas DateTime columns.

- **Initial implementation of LpScaler.**

    [PR#253](https://github.com/microsoft/NimbusML/pull/253)
    Normalize vectors (rows) individually by rescaling them to unit norm (L2, L1 or LInf).
    Performs the following operation on a vector X: Y = (X - M) / D, where M is mean and D
    is either L2 norm, L1 norm or LInf norm.

- **Add support for variable length vector output.**

    [PR#267](https://github.com/microsoft/NimbusML/pull/267)
    Support output of columns returned from ML.Net which contain variable length vectors.

- **Save `predictor_model` when pickling a `Pipeline`.**

    [PR#295](https://github.com/microsoft/NimbusML/pull/295)

- **Initial implementation of the WordTokenizer transform.**

    [PR#296](https://github.com/microsoft/NimbusML/pull/296)

- **Add support for summary output from tree based predictors.**

    [PR#298](https://github.com/microsoft/NimbusML/pull/298)

## **Bug Fixes**

- **Fixed `Pipeline.transform()` in transform only `Pipeline` fails if y column is provided **

    [PR#294](https://github.com/microsoft/NimbusML/pull/294)
    Enable calling `.transform()` on a `Pipeline` containing only transforms when the y column is provided 

- **Fix issue when using `predict_proba` or `decision_function` with combined models.**

    [PR#272](https://github.com/microsoft/NimbusML/pull/272)

- **Fix `Pipeline._extract_classes_from_headers` was not checking for valid steps.**

    [PR#292](https://github.com/microsoft/NimbusML/pull/292)

- **Fix BinaryDataStream was not valid as input for transformer.**

    [PR#307](https://github.com/microsoft/NimbusML/pull/307)

- **Fix casing for the installPythonPackages build.sh argument.**

    [PR#256](https://github.com/microsoft/NimbusML/pull/256)

## **Breaking Changes**

- **Removed `y` parameter from `Pipeline.transform()`**

    [PR#294](https://github.com/microsoft/NimbusML/pull/294)
    Removed `y` parameter from `Pipeline.transform()` as it is not needed nor used for transforming data with a fitted `Pipeline`.

## **Enhancements**

None.

## **Documentation and Samples**

None. 

## **Remarks**

None.
