# [NimbusML](https://docs.microsoft.com/en-us/nimbusml/overview) 1.2.0

## **New Features**

- **Time Series Spike And Change Point Detection**

   [PR#135](https://github.com/microsoft/NimbusML/pull/135) added support
   for time series spike and change point detection using the Independent
   and identically distributed (IID) and Singular Spectrum Analysis (SSA)
   algorithms.

   **Spike Detection Examples**

   - [IID using a pandas Series](https://github.com/microsoft/NimbusML/tree/master/src/python/nimbusml/examples/examples_from_dataframe/IidSpikeDetector_df.py)
   - [SSA using a pandas Series](https://github.com/microsoft/NimbusML/tree/master/src/python/nimbusml/examples/examples_from_dataframe/SsaSpikeDetector_df.py)
   - [IID using a FileDataStream](https://github.com/microsoft/NimbusML/tree/master/src/python/nimbusml/examples/IidSpikeDetector.py)
   - [SSA using a FileDataStream](https://github.com/microsoft/NimbusML/tree/master/src/python/nimbusml/examples/SsaSpikeDetector.py)

   **Change Point Detection Examples**

   - [IID using a pandas Series](https://github.com/microsoft/NimbusML/tree/master/src/python/nimbusml/examples/examples_from_dataframe/IidChangePointDetector_df.py)
   - [SSA using a pandas Series](https://github.com/microsoft/NimbusML/tree/master/src/python/nimbusml/examples/examples_from_dataframe/SsaChangePointDetector_df.py)
   - [IID using a FileDataStream](https://github.com/microsoft/NimbusML/tree/master/src/python/nimbusml/examples/IidChangePointDetector.py)
   - [SSA using a FileDataStream](https://github.com/microsoft/NimbusML/tree/master/src/python/nimbusml/examples/SsaChangePointDetector.py)

- **Time Series Forecasting**
   [PR#164](https://github.com/microsoft/NimbusML/pull/164) exposes an API
   for time series forecasting using Singular Spectrum Analysis(SSA).

   **Forecasting Examples**
   - [Using a pandas Series](https://github.com/microsoft/NimbusML/tree/master/src/python/nimbusml/examples/examples_from_dataframe/SsaForecaster_df.py)
   - [Using a FileDataStream](https://github.com/microsoft/NimbusML/tree/master/src/python/nimbusml/examples/SsaForecaster.py)

## **Bug Fixes**

None.

## **Breaking Changes**

None.

## **Enhancements**

None.

## **Documentation and Samples**

- Sample for CharTokenizer. [PR#153](https://github.com/microsoft/NimbusML/pull/153)

## **Remarks**

None.
