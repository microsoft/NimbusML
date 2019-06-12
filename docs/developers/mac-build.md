Building NimbusML from source on Mac
==========================================
## Prerequisites
1. Xcode Command Line Tools (for Clang compiler)
2. cmake

## Build
Run `./build.sh`

This downloads dependencies (.NET SDK, specific versions of Python and Boost), builds native code and managed code, and packages NimbusML into a pip-installable wheel. This produces debug binaries by default, and release versions can be specified by `./build.sh --configuration RlsMacPy3.7` for examle.

For additional options including running tests and building components independently, see `./build.sh -h`.

### Notes
The LightGBM estimator currently has a runtime dependency on Gnu OpenMP libs on Mac. These can be obtained by installing gcc 4.2 or later, which can be done through homebrew with:
```brew gcc``` 
Running LightGBM without this will give the following error: 
```System.DllNotFoundException: 'Unable to load DLL 'lib_lightgbm': The specified module or one of its dependencies could not be found.'```