#!/usr/bin/env bash
set -e

ProductVersion=$(<version.txt)

# Store current script directory
__currentScriptDir=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
BuildOutputDir=${__currentScriptDir}/x64/
DependenciesDir=${__currentScriptDir}/dependencies
mkdir -p "${DependenciesDir}"

usage()
{
    echo "Usage: $0 --configuration <Configuration> [--runTests]"
    echo ""
    echo "Options:"
    echo "  --configuration <Configuration>   Build Configuration (DbgLinPy3.6,DbgLinPy3.5,DbgLinPy2.7,RlsLinPy3.6,RlsLinPy3.5,RlsLinPy2.7,DbgMacPy3.6,DbgMacPy3.5,DbgMacPy2.7,RlsMacPy3.6,RlsMacPy3.5,RlsMacPy2.7)"
    echo "  --runTests                        Run tests after build"
    echo "  --runTestsOnly                    Run tests on a wheel file in default build location (<repo>/target/)"
    echo "  --buildNativeBridgeOnly           Build only the native bridge code"
    echo "  --skipNativeBridge                Build the DotNet bridge and python wheel but use existing native bridge binaries (e.g. <repo>/x64/DbgLinPy3.6/pybridge.so)"
    exit 1
}

__configuration=DbgLinPy3.6
__runTests=false
__buildNativeBridge=true
__buildDotNetBridge=true

while [ "$1" != "" ]; do
    lowerI="$(echo $1 | awk '{print tolower($0)}')"
    case $lowerI in
        -h|--help)
            usage
            exit 1
            ;;
        --configuration)
            shift
            __configuration=$1
            ;;
        --runtests)
            __runTests=true
            ;;
        --runtestsonly)
            __buildNativeBridge=false
            __buildDotNetBridge=false
            __runTests=true
            ;;
        --buildnativebridgeonly)
            __buildDotNetBridge=false
            ;;
        --skipnativebridge)
            __buildNativeBridge=false
            ;;
        *)
        echo "Unknown argument to build.sh $1"; usage; exit 1
    esac
    shift
done

case $__configuration in
*LinPy3.7)
    PythonUrl=https://pythonpkgdeps.blob.core.windows.net/anaconda-full/Anaconda3-Linux-5.3.tar.gz
    PythonVersion=3.7
    PythonTag=cp37
    USE_PYBIND11=true
    ;;
*LinPy3.6)
    PythonUrl=https://pythonpkgdeps.blob.core.windows.net/anaconda-full/Anaconda3-Linux-5.0.1.v2.tar.gz
    PythonVersion=3.6
    PythonTag=cp36
    USE_PYBIND11=true
    ;;
*LinPy3.5)
    PythonUrl=https://pythonpkgdeps.blob.core.windows.net/anaconda-full/Anaconda3-Linux-4.2.0.v9.tar.gz
    BoostUrl=https://pythonpkgdeps.blob.core.windows.net/boost/release/linux/Boost-3.5-1.64.0.0.tar.gz 
    PythonVersion=3.5
    PythonTag=cp35
    USE_PYBIND11=false
    ;;
*LinPy2.7)
    PythonUrl=https://pythonpkgdeps.blob.core.windows.net/anaconda-full/Anaconda2-Linux-5.0.1.v2.tar.gz
    BoostUrl=https://pythonpkgdeps.blob.core.windows.net/boost/release/linux/Boost-2.7-1.64.0.0.tar.gz
    PythonVersion=2.7
    PythonTag=cp27
    USE_PYBIND11=false
    ;;
*MacPy3.7)
    PythonUrl=https://pythonpkgdeps.blob.core.windows.net/anaconda-full/Anaconda3-Mac-5.3.tar.gz
    PythonVersion=3.7
    PythonTag=cp37
    USE_PYBIND11=true
    ;;
*MacPy3.6)
    PythonUrl=https://pythonpkgdeps.blob.core.windows.net/anaconda-full/Anaconda3-Mac-5.0.1.tar.gz
    PythonVersion=3.6
    PythonTag=cp36
    USE_PYBIND11=true
    ;;
*MacPy3.5)
    PythonUrl=https://pythonpkgdeps.blob.core.windows.net/anaconda-full/Anaconda3-Mac-4.2.0.tar.gz
    BoostUrl=https://pythonpkgdeps.blob.core.windows.net/boost/release/mac/Boost-3.5-1.64.0.0.tar.gz 
    PythonVersion=3.5
    PythonTag=cp35
    USE_PYBIND11=false
    ;;
*MacPy2.7)
    PythonUrl=https://pythonpkgdeps.blob.core.windows.net/anaconda-full/Anaconda2-Mac-5.0.2.tar.gz
    BoostUrl=https://pythonpkgdeps.blob.core.windows.net/boost/release/mac/Boost-2.7-1.64.0.0.tar.gz
    PythonVersion=2.7
    PythonTag=cp27
    USE_PYBIND11=false
    ;;
esac

PythonRoot=${DependenciesDir}/Python${PythonVersion}
BoostRoot=${DependenciesDir}/Boost${PythonVersion}
# Platform name for python wheel based on OS
PlatName=manylinux1_x86_64
if [ "$(uname -s)" = "Darwin" ]
then 
    PlatName=macosx_10_11_x86_64
fi

echo ""
echo "#################################"
echo "Downloading Dependencies "
echo "#################################"
# Download & unzip Python
if [ ! -e "${PythonRoot}/.done" ]
then
    mkdir -p "${PythonRoot}"
    echo "Downloading and extracting Python archive ... "
    curl "${PythonUrl}" | tar xz -C "${PythonRoot}"
    # Move all binaries out of "anaconda3", "anaconda2", or "anaconda", depending on naming convention for version
    mv "${PythonRoot}/anaconda"*/* "${PythonRoot}/"
    touch "${PythonRoot}/.done"
fi
PythonExe="${PythonRoot}/bin/python"
echo "Python executable: ${PythonExe}"

# Download & unzip Boost or pybind11
if [ ${USE_PYBIND11} = true ]
    if [ ! -e "${BoostRoot}/.done" ]
    then
        echo "Instaling boost_python..."
        mkdir -p "${BoostRoot}"
        echo "Downloading and extracting Boost archive ... "
        curl "${BoostUrl}" | tar xz -C "${BoostRoot}"
        touch "${BoostRoot}/.done"
    fi
else
    echo "Installing pybind11 ..." 
    "${PythonExe}" -m pip install pybind11 
fi    

if [ ${__buildNativeBridge} = true ]
then 
    echo "Building Native Bridge ... "
    bash "${__currentScriptDir}/src/NativeBridge/build.sh" --configuration $__configuration --pythonver "${PythonVersion}" --pythonpath "${PythonRoot}" --boostpath "${BoostRoot}" 
fi

if [ ${__buildDotNetBridge} = true ]
then 
    # Install dotnet SDK version, see https://docs.microsoft.com/en-us/dotnet/core/tools/dotnet-install-script
    echo "Installing dotnet SDK ... "
    curl -sSL https://dot.net/v1/dotnet-install.sh | bash /dev/stdin -Version 2.1.200 -InstallDir ./cli

    # Build managed code
    echo "Building managed code ... "
    _dotnet="${__currentScriptDir}/cli/dotnet"
    ${_dotnet} build -c ${__configuration} --force "${__currentScriptDir}/src/Platforms/build.csproj"
    PublishDir=linux-x64
    if [ "$(uname -s)" = "Darwin" ]
    then 
        PublishDir=osx-x64
    fi
    ${_dotnet} publish "${__currentScriptDir}/src/Platforms/build.csproj" --force --self-contained -r ${PublishDir} -c ${__configuration}
    ${_dotnet} build -c ${__configuration} -o "${BuildOutputDir}/${__configuration}"  --force "${__currentScriptDir}/src/DotNetBridge/DotNetBridge.csproj"

    # Build nimbusml wheel
    echo ""
    echo "#################################"
    echo "Building nimbusml wheel package ... "
    echo "#################################"
    # Clean out build, dist, and libs from previous builds
    build="${__currentScriptDir}/src/python/build"
    dist="${__currentScriptDir}/src/python/dist"
    libs="${__currentScriptDir}/src/python/nimbusml/internal/libs"
    rm -rf "${build}"
    rm -rf "${dist}"
    rm -rf "${libs}"
    mkdir -p "${libs}"
    touch "${__currentScriptDir}/src/python/nimbusml/internal/libs/__init__.py"

    echo "Placing binaries in libs dir for wheel packaging ... "
    cp  "${BuildOutputDir}/${__configuration}"/DotNetBridge.dll "${__currentScriptDir}/src/python/nimbusml/internal/libs/"
    cp  "${BuildOutputDir}/${__configuration}"/pybridge.so "${__currentScriptDir}/src/python/nimbusml/internal/libs/"

    if [ ${PythonVersion} = 2.7 ]
    then
        cp  "${BuildOutputDir}/${__configuration}/Platform/${PublishDir}"/publish/*.dll "${__currentScriptDir}/src/python/nimbusml/internal/libs/"
        cp  "${BuildOutputDir}/${__configuration}/Platform/${PublishDir}"/publish/System.Native.a "${__currentScriptDir}/src/python/nimbusml/internal/libs/"
        cp  "${BuildOutputDir}/${__configuration}/Platform/${PublishDir}"/publish/createdump "${__currentScriptDir}/src/python/nimbusml/internal/libs/"  || :
        cp  "${BuildOutputDir}/${__configuration}/Platform/${PublishDir}"/publish/sosdocsunix.txt "${__currentScriptDir}/src/python/nimbusml/internal/libs/"
		ext=*.so
		if [ "$(uname -s)" = "Darwin" ]
		then 
            ext=*.dylib
		fi	
		cp  "${BuildOutputDir}/${__configuration}/Platform/${PublishDir}"/publish/${ext} "${__currentScriptDir}/src/python/nimbusml/internal/libs/"
    else
		libs_txt=libs_linux.txt
		if [ "$(uname -s)" = "Darwin" ]
		then 
		    libs_txt=libs_mac.txt
		fi
		cat build/${libs_txt} | while read i; do
			cp  "${BuildOutputDir}/${__configuration}/Platform/${PublishDir}"/publish/$i "${__currentScriptDir}/src/python/nimbusml/internal/libs/"
		done
    fi
	
    if [[ $__configuration = Dbg* ]]
    then
        cp  "${BuildOutputDir}/${__configuration}"/DotNetBridge.pdb "${__currentScriptDir}/src/python/nimbusml/internal/libs/"
    fi
  
    "${PythonExe}" -m pip install --upgrade "wheel>=0.31.0"
    cd "${__currentScriptDir}/src/python"

    "${PythonExe}" setup.py bdist_wheel --python-tag ${PythonTag} --plat-name ${PlatName}
    cd "${__currentScriptDir}"

    WheelFile=nimbusml-${ProductVersion}-${PythonTag}-none-${PlatName}.whl
    if [ ! -e "${__currentScriptDir}/src/python/dist/${WheelFile}" ]
    then
        echo "setup.py did not produce expected ${WheelFile}"
        exit 1
    fi

    rm -rf "${__currentScriptDir}/target"
    mkdir -p "${__currentScriptDir}/target"
    mv "${__currentScriptDir}/src/python/dist/${WheelFile}" "${__currentScriptDir}/target/"
    echo Python package successfully created: ${__currentScriptDir}/target/${WheelFile}
fi

if [ ${__runTests} = true ]
then 
    echo ""
    echo "#################################"
    echo "Running tests ... "
    echo "#################################"
    Wheel=${__currentScriptDir}/target/nimbusml-${ProductVersion}-${PythonTag}-none-${PlatName}.whl
    if [ ! -f ${Wheel} ]
    then
        echo "Unable to find ${Wheel}"
        exit 1
    fi
    # Review: Adding "--upgrade" to pip install will cause problems when using Anaconda as the python distro because of Anaconda's quirks with pytest.
    "${PythonExe}" -m pip install nose pytest graphviz pytest-cov "jupyter_client>=4.4.0" "nbconvert>=4.2.0"
    if [ ${PythonVersion} = 2.7 ]
    then
        "${PythonExe}" -m pip install --upgrade pyzmq
    fi
    "${PythonExe}" -m pip install --upgrade "${Wheel}"
    "${PythonExe}" -m pip install "scikit-learn==0.19.2"

    PackagePath=${PythonRoot}/lib/python${PythonVersion}/site-packages/nimbusml
    TestsPath1=${PackagePath}/tests
    TestsPath2=${__currentScriptDir}/src/python/tests
    ReportPath=${__currentScriptDir}/build/TestCoverageReport
    "${PythonExe}" -m pytest --verbose --maxfail=1000 --capture=sys "${TestsPath1}" --cov="${PackagePath}" --cov-report term-missing --cov-report html:"${ReportPath}"
    "${PythonExe}" -m pytest --verbose --maxfail=1000 --capture=sys "${TestsPath2}" --cov="${PackagePath}" --cov-report term-missing --cov-report html:"${ReportPath}"
fi

exit $?
