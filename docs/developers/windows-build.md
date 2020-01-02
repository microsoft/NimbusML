Building NimbusML from source on Windows
==========================================
## Prerequisites
1. Visual Studio 2015 or higher
    - Select C++ development tools from installer

## Build
Run `build.cmd`

This downloads dependencies (.NET SDK, specific versions of Python and Boost), builds native code and managed code, and packages NimbusML into a pip-installable wheel. This produces debug binaries by default, and release versions can be specified by `build.cmd --configuration RlsWinPy3.7` for example.

For additional options including running tests and building components independently, see `build.cmd -?`.
