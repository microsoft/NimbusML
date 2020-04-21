Building NimbusML from source on Linux
==========================================
## Prerequisites
1. gcc >= 5.4
2. cmake
3. curl
4. libunwind8*
5. libicu*

    *These are already included in most distros. If you need them and you have trouble finding them in your package repo, they can be gathered by installing the [.NET SDK](https://www.microsoft.com/net/download).

## Build
Run `./build.sh`

This downloads dependencies (.NET SDK, specific versions of Python and Boost), builds native code and managed code, and packages NimbusML into a pip-installable wheel. This produces debug binaries by default, and release versions can be specified by `./build.sh --configuration RlsLinPy3.7` for example.

For additional options including running tests and building components independently, see `./build.sh -h`.

### Known Issues
The LightGBM estimator fails on Linux when building from source. The official NimbusML Linux wheel package on Pypi.org has a working version of LightGBM.
