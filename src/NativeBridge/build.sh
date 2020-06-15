#!/usr/bin/env bash

set -e

usage()
{
    echo "Usage: $0 --configuration <Configuration> "
    echo ""
    echo "Options:"
    echo "  --configuration <Configuration>   Build Configuration (DbgLinPy3.8,DbgLinPy3.7,DbgLinPy3.6,RlsLinPy3.8,RlsLinPy3.7,RlsLinPy3.6,DbgMacPy3.8,DbgMacPy3.7,DbgMacPy3.6,RlsMacPy3.8,RlsMacPy3.7,RlsMacPy3.6)"
    echo "  --pythonver <Python version>      Python version number (3.8, 3.7, 3.6)"
    echo "  --pythonpath <Python path>        Path to python library."
    exit 1
}

SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ "$SOURCE" != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
RootRepo="$DIR/../.."

__configuration=DbgLinPy3.7
__pythonver=3.7
__rootBinPath="$RootRepo/x64"
__pythonpath=""

while [ "$1" != "" ]; do
        lowerI="$(echo $1 | awk '{print tolower($0)}')"
        case $lowerI in
        -h|--help)
            usage
            exit 1
            ;;
        --arch)
            shift
            __build_arch=$1
            ;;
        --configuration)
            shift
            __configuration=$1
            ;;
        --pythonver)
            shift
            __pythonver=$1
            ;;
        --pythonpath)
            shift
            __pythonpath=$1
            ;;
        *)
        echo "Unknown argument to build.sh $1"; usage; exit 1
    esac
    shift
done

# Set up the environment to be used for building with clang.
if command -v "clang-3.5" > /dev/null 2>&1; then	
    export CC="$(command -v clang-3.5)"	
    export CXX="$(command -v clang++-3.5)"	
elif command -v "clang-3.6" > /dev/null 2>&1; then	
    export CC="$(command -v clang-3.6)"	
    export CXX="$(command -v clang++-3.6)"	
elif command -v "clang-3.9" > /dev/null 2>&1; then	
    export CC="$(command -v clang-3.9)"	
    export CXX="$(command -v clang++-3.9)"	
elif command -v clang > /dev/null 2>&1; then	
    export CC="$(command -v clang)"	
    export CXX="$(command -v clang++)"	
else	
    echo "Unable to find Clang Compiler"	
    echo "Instal clang-3.5 or clang3.6 or clang3.9"	
    echo "Using default system compiler ... "	
fi

__cmake_defines="-DCMAKE_BUILD_TYPE=${__configuration} -DPYTHON_VER=${__pythonver} -DPYTHON_DIR=${__pythonpath}"

__IntermediatesDir="$__rootBinPath/$__configuration/obj"
rm -rf "$__IntermediatesDir"
mkdir -p "$__IntermediatesDir"
cd "$__IntermediatesDir"

echo "Building mlnet native components from $DIR to $(pwd)"
set -x # turn on trace
cmake "$DIR" -G "Unix Makefiles" $__cmake_defines
set +x # turn off trace
make install
