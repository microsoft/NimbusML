#!/usr/bin/env bash
set -e

ProductVersion=$(<version.txt)

# Store current script directory
__currentScriptDir=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
BuildOutputDir=${__currentScriptDir}/x64/

# Parameter defaults
__configuration=RlsLinPy3.7


PythonUrl=
BoostUrl=
PythonVersion=3.7
PythonTag=cp37
USE_PYBIND11=true

echo "Unknown configuration '$__configuration'"; usage; exit 1
esac

BoostRoot=

# Platform name for python wheel based on OS
PlatName=manylinux1_x86_64

echo ""
echo "#################################"
echo "Settings up "
echo "#################################"
# Download & unzip Python
PythonRoot=
PythonExe=python

echo "PythonRoot: ${PythonRoot}"
echo "PythonExe: ${PythonExe}"
echo "BoostRoot: ${BoostRoot}"

# Download & unzip Boost or pybind11
echo "Installing pybind11 ..." 
"${PythonExe}" -m pip install pybind11
echo "Done."

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

    libs_txt=libs_linux.txt
    cat build/${libs_txt} | while read i; do
        cp  "${BuildOutputDir}/${__configuration}/Platform/${PublishDir}"/publish/$i "${__currentScriptDir}/src/python/nimbusml/internal/libs/"
    done
	
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
    echo "PythonRoot=${PythonRoot}"
    echo "PythonExe=${PythonExe}"
    Wheel=${__currentScriptDir}/target/nimbusml-${ProductVersion}-${PythonTag}-none-${PlatName}.whl
    if [ ! -f ${Wheel} ]
    then
        echo "Unable to find ${Wheel}"
        exit 1
    fi
    # Review: Adding "--upgrade" to pip install will cause problems when using Anaconda as the python distro because of Anaconda's quirks with pytest.
    "${PythonExe}" -m pip install nose pytest>=4.4.0 graphviz pytest-cov>=2.6.1 "jupyter_client>=4.4.0" "nbconvert>=4.2.0"
    if [ ${PythonVersion} = 2.7 ]
    then
        "${PythonExe}" -m pip install --upgrade pyzmq
    fi
    "${PythonExe}" -m pip install --upgrade "${Wheel}"
    "${PythonExe}" -m pip install "scikit-learn>=0.19.2"

    PackagePath=${PythonRoot}/lib/python${PythonVersion}/site-packages/nimbusml
    TestsPath1=${PackagePath}/tests
    TestsPath2=${__currentScriptDir}/src/python/tests
    ReportPath=${__currentScriptDir}/build/TestCoverageReport
    "${PythonExe}" -m pytest --verbose --maxfail=1000 --capture=sys "${TestsPath1}" --cov="${PackagePath}" --cov-report term-missing --cov-report html:"${ReportPath}"
    "${PythonExe}" -m pytest --verbose --maxfail=1000 --capture=sys "${TestsPath2}" --cov="${PackagePath}" --cov-report term-missing --cov-report html:"${ReportPath}"
fi

exit $?
