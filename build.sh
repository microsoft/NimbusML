#!/usr/bin/env bash
set -e

ProductVersion=$(<version.txt)

# Store current script directory
__currentScriptDir=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
BuildOutputDir=${__currentScriptDir}/x64/
DependenciesDir=${__currentScriptDir}/dependencies
# Platform name for python wheel based on OS
PlatName=manylinux1_x86_64
if [ "$(uname -s)" = "Darwin" ]
then 
    PlatName=macosx_10_11_x86_64
fi
mkdir -p "${DependenciesDir}"

usage()
{
    echo "Usage: $0 --configuration <Configuration> [--runTests] [--includeExtendedTests] [--installPythonPackages]"
    echo ""
    echo "Options:"
    echo "  --configuration <Configuration>   Build Configuration (DbgLinPy3.8, DbgLinPy3.7,DbgLinPy3.6,RlsLinPy3.8,RlsLinPy3.7,RlsLinPy3.6,DbgMacPy3.8,DbgMacPy3.7,DbgMacPy3.6,RlsMacPy3.8,RlsMacPy3.7,RlsMacPy3.6)"
    echo "  --runTests                        Run tests after build"
    echo "  --installPythonPackages           Install python packages after build"
    echo "  --runTestsOnly                    Run tests on a wheel file in default build location (<repo>/target/)"
    echo "  --includeExtendedTests            Include the extended tests if the tests are run"
    echo "  --buildNativeBridgeOnly           Build only the native bridge code"
    echo "  --skipNativeBridge                Build the DotNet bridge and python wheel but use existing native bridge binaries (e.g. <repo>/x64/DbgLinPy3.7/pybridge.so)"
    exit 1
}

# Parameter defaults
if [ "$(uname -s)" = "Darwin" ]
then 
    __configuration=DbgMacPy3.8
else
    __configuration=DbgLinPy3.8
fi
__runTests=false
__installPythonPackages=false
__runExtendedTests=false
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
            __installPythonPackages=true
            ;;
        --installpythonpackages)
            __installPythonPackages=true
            ;;
        --includeextendedtests)
            __runExtendedTests=true
            ;;
        --runtestsonly)
            __buildNativeBridge=false
            __buildDotNetBridge=false
            __runTests=true
            __installPythonPackages=true
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
*LinPy3.8)
    PythonUrl=https://pythonpkgdeps.blob.core.windows.net/python/python-3.8.3-linux64.v2.tar.gz
    PythonVersion=3.8
    PythonTag=cp38
    ;;
*LinPy3.7)
    PythonUrl=https://pythonpkgdeps.blob.core.windows.net/anaconda-full/Anaconda3-Linux-2019.03.v2.tar.gz
    PythonVersion=3.7
    PythonTag=cp37
    ;;
*LinPy3.6)
    PythonUrl=https://pythonpkgdeps.blob.core.windows.net/anaconda-full/Anaconda3-Linux-5.0.1.v2.tar.gz
    PythonVersion=3.6
    PythonTag=cp36
    ;;
*MacPy3.8)
    PythonUrl=https://pythonpkgdeps.blob.core.windows.net/python/python-3.8.3-mac64.tar.gz
    PythonVersion=3.8
    PythonTag=cp38
    ;;
*MacPy3.7)
    PythonUrl=https://pythonpkgdeps.blob.core.windows.net/anaconda-full/Anaconda3-Mac-2019.03.v2.tar.gz
    PythonVersion=3.7
    PythonTag=cp37
    ;;
*MacPy3.6)
    PythonUrl=https://pythonpkgdeps.blob.core.windows.net/anaconda-full/Anaconda3-Mac-5.0.1.tar.gz
    PythonVersion=3.6
    PythonTag=cp36
    ;;
*)
echo "Unknown configuration '$__configuration'"; usage; exit 1
esac

PythonRoot=${DependenciesDir}/Python${PythonVersion}
echo "Python root: ${PythonRoot}"
PythonExe="${PythonRoot}/bin/python"
if [ ${PythonVersion} = 3.8 ] && [ "$(uname -s)" != "Darwin" ]
then 
    PythonExe="python3.8" # use prebuilt version of in docker image on Linux
fi
echo "Python executable: ${PythonExe}"

echo "Downloading Python Dependencies "
# Download & unzip Python
if [ ! -e "${PythonRoot}/.done" ]
then
    mkdir -p "${PythonRoot}"
    echo "Downloading and extracting Python archive ... "
    curl "${PythonUrl}" | tar xz -C "${PythonRoot}"
    if [ ${PythonVersion} != 3.8 ]
    then 
        # Move all binaries out of "anaconda3", "anaconda2", or "anaconda", depending on naming convention for version
        mv "${PythonRoot}/anaconda"*/* "${PythonRoot}/"
        touch "${PythonRoot}/.done"
        echo "Install libc6-dev ... "
        if [ "$(uname -s)" != "Darwin" ]
        then 
            {
                apt-get update 
                # Required for Image.py and Image_df.py to run successfully on Ubuntu.
                apt-get install libc6-dev -y
                #apt-get install libgdiplus -y
                # Required for onnxruntime tests
                #apt-get install -y locales
                #locale-gen en_US.UTF-8
            } || { 
                yum update --skip-broken
                # Required for Image.py and Image_df.py to run successfully on CentOS.
                yum install glibc-devel -y
                # Required for onnxruntime tests
                # yum install glibc-all-langpacks
                # localedef -v -c -i en_US -f UTF-8 en_US.UTF-8
            }
        fi
        echo "Install pybind11 ... "
        "${PythonRoot}/bin/python" -m pip install pybind11
        echo "Done installing pybind11 ... "
    fi
    touch "${PythonRoot}/.done"
fi

if [ ${__buildNativeBridge} = true ]
then 
    echo "Building Native Bridge ... "
    bash "${__currentScriptDir}/src/NativeBridge/build.sh" --configuration $__configuration --pythonver "${PythonVersion}" --pythonpath "${PythonRoot}"
    rm -rf "${__currentScriptDir}/src/NativeBridge/x64"
fi

if [ ${__buildDotNetBridge} = true ]
then 
    # Install dotnet SDK version, see https://docs.microsoft.com/en-us/dotnet/core/tools/dotnet-install-script
    echo "Installing dotnet SDK ... "
    curl -sSL https://dot.net/v1/dotnet-install.sh | bash /dev/stdin -Version 3.1.102 -InstallDir ./cli

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
    mv  "${BuildOutputDir}/${__configuration}"/DotNetBridge.dll "${__currentScriptDir}/src/python/nimbusml/internal/libs/"
    mv  "${BuildOutputDir}/${__configuration}"/pybridge.so "${__currentScriptDir}/src/python/nimbusml/internal/libs/"

    # ls -l "${BuildOutputDir}/${__configuration}/Platform/${PublishDir}"/publish/
    libs_txt=libs_linux.txt
    if [ "$(uname -s)" = "Darwin" ]
    then 
        libs_txt=libs_mac.txt
    fi
    cat build/${libs_txt} | while read i; do
        mv  "${BuildOutputDir}/${__configuration}/Platform/${PublishDir}"/publish/$i "${__currentScriptDir}/src/python/nimbusml/internal/libs/"
    done
    mv  "${BuildOutputDir}/${__configuration}/Platform/${PublishDir}"/publish/Data "${__currentScriptDir}/src/python/nimbusml/internal/libs/."
    
    if [[ $__configuration = Dbg* ]]
    then
        mv  "${BuildOutputDir}/${__configuration}"/DotNetBridge.pdb "${__currentScriptDir}/src/python/nimbusml/internal/libs/"
    fi
  
    # Clean out space for building wheel
    echo "Deleting ${BuildOutputDir} ${__currentScriptDir}/cli"
    rm -rf "${BuildOutputDir}"
    rm -rf "${__currentScriptDir}/cli"

    cd "${__currentScriptDir}/src/python"
    if [ ${PythonVersion} = 3.8 ]
    then 
        # this is actually python 3.6 preinstalled, it can do 3.8 package
        python3 setup.py bdist_wheel --python-tag ${PythonTag} --plat-name ${PlatName}
    else
        "${PythonExe}" -m pip install "wheel>=0.31.0"
        "${PythonExe}" setup.py bdist_wheel --python-tag ${PythonTag} --plat-name ${PlatName}
    fi
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
    echo "Deleting ${build} ${dist} ${libs} ... "
    rm -rf "${build}"
    rm -rf "${dist}"
    rm -rf "${libs}"
fi

if [ ${__installPythonPackages} = true ]
then
    echo ""
    echo "#################################"
    echo "Installing Python packages ... "
    echo "#################################"
    Wheel=${__currentScriptDir}/target/nimbusml-${ProductVersion}-${PythonTag}-none-${PlatName}.whl
    if [ ! -f ${Wheel} ]
    then
        echo "Unable to find ${Wheel}"
        exit 1
    fi
    if [ ${PythonVersion} = 3.8 ] && [ "$(uname -s)" = "Darwin" ]
    then
        echo "Installing python 3.8 on Mac ... "
        curl -O https://www.python.org/ftp/python/3.8.3/python-3.8.3-macosx10.9.pkg
        sudo installer -pkg python-3.8.3-macosx10.9.pkg -target /
    fi

    if [ ${PythonVersion} = 3.8 ] && [ "$(uname -s)" != "Darwin" ]
    then
        "${PythonExe}" -m pip install --user nose "pytest>=4.4.0" pytest-xdist graphviz
        "${PythonExe}" -m pip install --user --upgrade "azureml-dataprep>=1.1.33"
        "${PythonExe}" -m pip install --user --upgrade onnxruntime
        "${PythonExe}" -m pip install --user --upgrade "${Wheel}"
        "${PythonExe}" -m pip install --user scipy "scikit-learn==0.19.2"
    else
        # Review: Adding "--upgrade" to pip install will cause problems when using Anaconda as the python distro because of Anaconda's quirks with pytest.
        "${PythonExe}" -m pip install nose "pytest>=4.4.0" pytest-xdist graphviz
        "${PythonExe}" -m pip install --upgrade "azureml-dataprep>=1.1.33"
        "${PythonExe}" -m pip install --upgrade onnxruntime
        "${PythonExe}" -m pip install --upgrade "${Wheel}"
        "${PythonExe}" -m pip install "scikit-learn==0.19.2"
    fi
    if [ ${PythonVersion} = 3.6 ] && [ "$(uname -s)" = "Darwin" ]
    then
        "${PythonExe}" -m pip install --upgrade pytest-remotedata
    fi

fi

if [ ${__runTests} = true ]
then 
    echo ""
    echo "#################################"
    echo "Running tests ... "
    echo "#################################"
    PackagePath=${PythonRoot}/lib/python${PythonVersion}/site-packages/nimbusml
    TestsPath1=${PackagePath}/tests
    TestsPath2=${__currentScriptDir}/src/python/tests
    TestsPath3=${__currentScriptDir}/src/python/tests_extended
    if [  ${PythonVersion} = 3.8 ]
    then
        if [ "$(uname -s)" = "Darwin" ]
        then
            TestsPath1=/Library/Frameworks/Python.framework/Versions/3.8/lib/python3.8/site-packages/nimbusml/tests
        else
            # Linux Python3.8 only here.
            TestsPath1=/home/runner/.local/lib/python3.8/site-packages/nimbusml/tests
        fi
        echo "Test paths: ${TestsPath1} ${TestsPath2} "
        "${PythonExe}" -m pytest -n 4 --verbose --maxfail=1000 --capture=sys "${TestsPath2}" "${TestsPath1}" 
    else
        "${PythonExe}" -m pytest -n 4 --verbose --maxfail=1000 --capture=sys "${TestsPath2}" "${TestsPath1}" || \
            "${PythonExe}" -m pytest -n 4 --last-failed --verbose --maxfail=1000 --capture=sys "${TestsPath2}" "${TestsPath1}" 
    fi

    if [ ${__runExtendedTests} = true ]
    then
        echo "Running extended tests ... " 
        if [ "$(uname -s)" != "Darwin" ]
        then 
            {
                apt-get update 
                # Required for Image.py and Image_df.py to run successfully on Ubuntu.
                apt-get install libc6-dev -y
                apt-get install libgdiplus -y
                # Required for onnxruntime tests
                apt-get install -y locales
                locale-gen en_US.UTF-8
            } || { 
                yum update --skip-broken
                # Required for Image.py and Image_df.py to run successfully on CentOS.
                yum install glibc-devel -y
                # Required for onnxruntime tests
                yum install glibc-all-langpacks
                localedef -v -c -i en_US -f UTF-8 en_US.UTF-8
            }
        fi
        "${PythonExe}" -m pytest -n 4 --verbose --maxfail=1000 --capture=sys "${TestsPath3}"
    fi
fi

exit $?
