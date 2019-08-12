@if not defined _echo @echo off
setlocal

set /p ProductVersion=<version.txt

:: Store current script directory before %~dp0 gets affected by another process later.
set __currentScriptDir=%~dp0
set DependenciesDir=%__currentScriptDir%dependencies\
if not exist "%DependenciesDir%" (md "%DependenciesDir%")

:: Default configuration if no arguents passed to build.cmd (DbgWinPy3.7)
set __BuildArch=x64
set __VCBuildArch=x86_amd64
set Configuration=DbgWinPy3.7
set DebugBuild=True
set BuildOutputDir=%__currentScriptDir%x64\
set PythonUrl=https://pythonpkgdeps.blob.core.windows.net/python/python-3.7.3-amd64.zip
set PythonRoot=%DependenciesDir%Python3.7
set BoostUrl=https://pythonpkgdeps.blob.core.windows.net/boost/debug/windows/Boost-3.7-1.69.0.0.zip
set BoostRoot=%DependenciesDir%BoostDbg3.7
set PythonVersion=3.7
set PythonTag=cp37
set RunTests=False
set InstallPythonPackages=False
set RunExtendedTests=False
set BuildDotNetBridgeOnly=False
set SkipDotNetBridge=False
set AzureBuild=False

:Arg_Loop
if [%1] == [] goto :Build
if /i [%1] == [--configuration] (
    shift && goto :Configuration
)
if /i [%1] == [--runTests]     (
    set RunTests=True
    set InstallPythonPackages=True
    shift && goto :Arg_Loop
)
if /i [%1] == [--installPythonPackages]     (
    set InstallPythonPackages=True
    shift && goto :Arg_Loop
)
if /i [%1] == [--includeExtendedTests]     (
    set RunExtendedTests=True
    shift && goto :Arg_Loop
)
if /i [%1] == [--buildDotNetBridgeOnly]     (
    set BuildDotNetBridgeOnly=True
    shift && goto :Arg_Loop
)
if /i [%1] == [--skipDotNetBridge]     (
    set SkipDotNetBridge=True
    shift && goto :Arg_Loop
)
if /i [%1] == [--azureBuild]     (
    set AzureBuild=True
    shift && goto :Arg_Loop
) else goto :Usage

:Usage
echo "Usage: build.cmd [--configuration <Configuration>] [--runTests] [--includeExtendedTests] [--buildDotNetBridgeOnly] [--skipDotNetBridge] [--azureBuild]"
echo ""
echo "Options:"
echo "  --configuration <Configuration>   Build Configuration (DbgWinPy3.7,DbgWinPy3.6,DbgWinPy3.5,DbgWinPy2.7,RlsWinPy3.7,RlsWinPy3.6,RlsWinPy3.5,RlsWinPy2.7)"
echo "  --runTests                        Run tests after build"
echo "  --installPythonPackages           Install python packages after build"
echo "  --includeExtendedTests            Include the extended tests if the tests are run"
echo "  --buildDotNetBridgeOnly           Build only DotNetBridge"
echo "  --skipDotNetBridge                Build everything except DotNetBridge"
echo "  --azureBuild                      Building in azure devops (adds dotnet CLI to the path)"
goto :Exit_Success

:Configuration
if /i [%1] == [RlsWinPy3.7]     (
    set DebugBuild=False
    set Configuration=RlsWinPy3.7
    set PythonUrl=https://pythonpkgdeps.blob.core.windows.net/python/python-3.7.3-amd64.zip
    set PythonRoot=%DependenciesDir%Python3.7
    set BoostUrl=https://pythonpkgdeps.blob.core.windows.net/boost/release/windows/Boost-3.7-1.69.0.0.zip
    set BoostRoot=%DependenciesDir%BoostRls3.7
    set PythonVersion=3.7
    set PythonTag=cp37
    shift && goto :Arg_Loop
)
if /i [%1] == [RlsWinPy3.6]     (
    set DebugBuild=False
    set Configuration=RlsWinPy3.6
    set PythonUrl=https://pythonpkgdeps.blob.core.windows.net/python/python-3.6.5-mohoov-amd64.zip
    set PythonRoot=%DependenciesDir%Python3.6
    set BoostUrl=https://pythonpkgdeps.blob.core.windows.net/boost/release/windows/Boost-3.6-1.64.0.0.zip
    set BoostRoot=%DependenciesDir%BoostRls3.6
    set PythonVersion=3.6
    set PythonTag=cp36
    shift && goto :Arg_Loop
)
if /i [%1] == [RlsWinPy3.5]     (
    set DebugBuild=False
    set Configuration=RlsWinPy3.5
    set PythonUrl=https://pythonpkgdeps.blob.core.windows.net/python/python-3.5.4-mohoov-amd64.zip
    set PythonRoot=%DependenciesDir%Python3.5
    set BoostUrl=https://pythonpkgdeps.blob.core.windows.net/boost/release/windows/Boost-3.5-1.64.0.0.zip
    set BoostRoot=%DependenciesDir%BoostRls3.5
    set PythonVersion=3.5
    set PythonTag=cp35
    shift && goto :Arg_Loop
)
if /i [%1] == [RlsWinPy2.7]     (
    set DebugBuild=False
    set Configuration=RlsWinPy2.7
    set PythonUrl=https://pythonpkgdeps.blob.core.windows.net/python/python-2.7.15-mohoov-amd64.zip
    set PythonRoot=%DependenciesDir%Python2.7
    set BoostUrl=https://pythonpkgdeps.blob.core.windows.net/boost/release/windows/Boost-2.7-1.64.0.0.zip
    set BoostRoot=%DependenciesDir%BoostRls2.7
    set PythonVersion=2.7
    set PythonTag=cp27
    shift && goto :Arg_Loop
)
if /i [%1] == [DbgWinPy3.7]     (
    set DebugBuild=True
    set Configuration=DbgWinPy3.7
    set PythonUrl=https://pythonpkgdeps.blob.core.windows.net/python/python-3.7.3-amd64.zip
    set PythonRoot=%DependenciesDir%Python3.7
    set BoostUrl=https://pythonpkgdeps.blob.core.windows.net/boost/debug/windows/Boost-3.7-1.69.0.0.zip
    set BoostRoot=%DependenciesDir%BoostDbg3.7
    set PythonVersion=3.7
    set PythonTag=cp37
    shift && goto :Arg_Loop
)
if /i [%1] == [DbgWinPy3.6]     (
    set DebugBuild=True
    set Configuration=DbgWinPy3.6
    set PythonUrl=https://pythonpkgdeps.blob.core.windows.net/python/python-3.6.5-mohoov-amd64.zip
    set PythonRoot=%DependenciesDir%Python3.6
    set BoostUrl=https://pythonpkgdeps.blob.core.windows.net/boost/debug/windows/Boost-3.6-1.64.0.0.zip
    set BoostRoot=%DependenciesDir%BoostDbg3.6
    set PythonVersion=3.6
    set PythonTag=cp36
    shift && goto :Arg_Loop
)
if /i [%1] == [DbgWinPy3.5]     (
    set DebugBuild=True
    set Configuration=DbgWinPy3.5
    set PythonUrl=https://pythonpkgdeps.blob.core.windows.net/python/python-3.5.4-mohoov-amd64.zip
    set PythonRoot=%DependenciesDir%Python3.5
    set BoostUrl=https://pythonpkgdeps.blob.core.windows.net/boost/debug/windows/Boost-3.5-1.64.0.0.zip
    set BoostRoot=%DependenciesDir%BoostDbg3.5
    set PythonVersion=3.5
    set PythonTag=cp35
    shift && goto :Arg_Loop
)
if /i [%1] == [DbgWinPy2.7]     (
    set DebugBuild=True
    set Configuration=DbgWinPy2.7
    set PythonUrl=https://pythonpkgdeps.blob.core.windows.net/python/python-2.7.15-mohoov-amd64.zip
    set PythonRoot=%DependenciesDir%Python2.7
    set BoostUrl=https://pythonpkgdeps.blob.core.windows.net/boost/debug/windows/Boost-2.7-1.64.0.0.zip
    set BoostRoot=%DependenciesDir%BoostDbg2.7
    set PythonVersion=2.7
    set PythonTag=cp27
    shift && goto :Arg_Loop
)

:Build
:: Install dotnet SDK version, see https://docs.microsoft.com/en-us/dotnet/core/tools/dotnet-install-script
echo Installing dotnet SDK ... 
powershell -NoProfile -ExecutionPolicy unrestricted -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; &([scriptblock]::Create((Invoke-WebRequest -useb 'https://dot.net/v1/dotnet-install.ps1'))) -Version 2.1.701 -InstallDir ./cli"

set _dotnetRoot=%__currentScriptDir%cli

if "%AzureBuild%" == "True" (
    :: Add dotnet CLI root to the PATH in azure devops agent
    echo ##vso[task.prependpath]%_dotnetRoot%
)

:: Build managed code
echo ""
echo "#################################"
echo "Building DotNet Bridge ... "
echo "#################################"
set _dotnet=%_dotnetRoot%\dotnet.exe

if "%SkipDotNetBridge%" == "False" ( 
    call "%_dotnet%" build -c %Configuration% -o "%BuildOutputDir%%Configuration%"  --force "%__currentScriptDir%src\DotNetBridge\DotNetBridge.csproj"
)
if "%BuildDotNetBridgeOnly%" == "True" ( 
    exit /b %ERRORLEVEL%
)
call "%_dotnet%" build -c %Configuration% --force "%__currentScriptDir%src\Platforms\build.csproj"
call "%_dotnet%" publish "%__currentScriptDir%src\Platforms\build.csproj" --force --self-contained -r win-x64 -c %Configuration%

echo ""
echo "#################################"
echo "Downloading Dependencies "
echo "#################################"
:: Download & unzip Python
if not exist "%PythonRoot%\.done" (
    md "%PythonRoot%"
    echo Downloading python zip ... 
    powershell -command "& {$wc = New-Object System.Net.WebClient; $wc.DownloadFile('%PythonUrl%', '%DependenciesDir%python.zip');}"
    echo Extracting python zip ... 
    powershell.exe -nologo -noprofile -command "& { Add-Type -A 'System.IO.Compression.FileSystem'; [IO.Compression.ZipFile]::ExtractToDirectory('%DependenciesDir%python.zip', '%PythonRoot%'); }"
    echo.>"%PythonRoot%\.done"
    del %DependenciesDir%python.zip
)
:: Download & unzip Boost
if not exist "%BoostRoot%\.done" (
    md "%BoostRoot%"
    echo Downloading boost zip ... 
    powershell -command "& {$wc = New-Object System.Net.WebClient; $wc.DownloadFile('%BoostUrl%', '%DependenciesDir%boost.zip');}"
    echo Extracting boost zip ... 
    powershell.exe -nologo -noprofile -command "& { Add-Type -A 'System.IO.Compression.FileSystem'; [IO.Compression.ZipFile]::ExtractToDirectory('%DependenciesDir%boost.zip', '%BoostRoot%'); }"
    echo.>"%BoostRoot%\.done"
    del %DependenciesDir%boost.zip
)

echo ""
echo "#################################"
echo "Building Native Bridge ... "
echo "#################################"
:: Setting native code build environment
echo Setting native build environment ...
set _VSWHERE="%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set vswhereOutputFile=vswhereOutput.tmp

if exist %_VSWHERE% (
  %_VSWHERE% -version "[15.0,16.0)" -latest -prerelease -property installationPath > %vswhereOutputFile%
  for /f "tokens=* delims=" %%i in (%vswhereOutputFile%) do set _VSCOMNTOOLS=%%i\Common7\Tools
  del %vswhereOutputFile%
)
if not exist "%_VSCOMNTOOLS%" set _VSCOMNTOOLS=%VS140COMNTOOLS%
if not exist "%_VSCOMNTOOLS%" goto :MissingVersion

set "VSCMD_START_DIR=%__currentScriptDir%"
call "%_VSCOMNTOOLS%\VsDevCmd.bat"

if "%VisualStudioVersion%"=="15.0" (
    goto :VS2017
) else if "%VisualStudioVersion%"=="14.0" (
    goto :VS2015
) else goto :MissingVersion

:MissingVersion
:: Can't find VS 2015 or 2017
echo Error: Visual Studio 2015 or 2017 required
echo        Please see https://github.com/dotnet/machinelearning/tree/master/Documentation for build instructions.
goto :Exit_Error

:VS2017
:: Setup vars for VS2017
set __PlatformToolset=v141
set __VSVersion=15 2017
if NOT "%__BuildArch%" == "arm64" (
    :: Set the environment for the native build
    call "%VS150COMNTOOLS%..\..\VC\Auxiliary\Build\vcvarsall.bat" %__VCBuildArch%
)
goto :NativeBridge

:VS2015
:: Setup vars for VS2015build
set __PlatformToolset=v140
set __VSVersion=14 2015
if NOT "%__BuildArch%" == "arm64" (
    :: Set the environment for the native build
    call "%VS140COMNTOOLS%..\..\VC\vcvarsall.bat" %__VCBuildArch%
)
goto :NativeBridge

:NativeBridge
:: Build NativeBridge.vcxproj
echo Building NativeBridge.vcxproj ...
set __msbuildArgs=/p:Platform=%__BuildArch% /p:PlatformToolset="%__PlatformToolset%"
call msbuild  "%__currentScriptDir%src\NativeBridge\NativeBridge.vcxproj"  /p:Configuration=%Configuration%  %__msbuildArgs%
if %errorlevel% neq 0 goto :Exit_Error


:: Build nimbusml wheel
echo ""
echo "#################################"
echo "Building nimbusml wheel package ... "
echo "#################################"
echo Building nimbusml wheel package ...
set PythonExe=%PythonRoot%\python.exe
echo Python executable: %PythonExe%
:: Clean out build, dist, and libs from previous builds
set build="%__currentScriptDir%src\python\build"
set dist="%__currentScriptDir%src\python\dist"
set libs="%__currentScriptDir%src\python\nimbusml\internal\libs"
if exist %build% rd %build% /S /Q
if exist %dist% rd %dist% /S /Q
if exist %libs% rd %libs% /S /Q
md %libs%
echo.>"%__currentScriptDir%src\python\nimbusml\internal\libs\__init__.py"

if %PythonVersion% == 3.7 (
    :: Running the check in one python is enough. Entrypoint compiler doesn't run in py2.7.
    echo Generating low-level Python API from mainifest.json ...
    call "%PythonExe%" -m pip install --upgrade autopep8 autoflake isort jinja2
    cd "%__currentScriptDir%src\python"
    call "%PythonExe%" tools\entrypoint_compiler.py --check_manual_changes 
    if errorlevel 1 (
        echo Codegen check failed. Try running tools/entrypoint_compiler.py --check_manual_changes to find the problem.
        goto :Exit_Error
    )
    cd "%__currentScriptDir%"
)

echo Placing binaries in libs dir for wheel packaging
copy  "%BuildOutputDir%%Configuration%\DotNetBridge.dll" "%__currentScriptDir%src\python\nimbusml\internal\libs\"
copy  "%BuildOutputDir%%Configuration%\pybridge.pyd" "%__currentScriptDir%src\python\nimbusml\internal\libs\"

if %PythonVersion% == 2.7 (
    copy "%BuildOutputDir%%Configuration%\Platform\win-x64\publish\*.dll" "%__currentScriptDir%src\python\nimbusml\internal\libs\"
) else (
    for /F "tokens=*" %%A in (build/libs_win.txt) do copy "%BuildOutputDir%%Configuration%\Platform\win-x64\publish\%%A" "%__currentScriptDir%src\python\nimbusml\internal\libs\"
)

if "%DebugBuild%" == "True" (
    copy  "%BuildOutputDir%%Configuration%\DotNetBridge.pdb" "%__currentScriptDir%src\python\nimbusml\internal\libs\"
    copy  "%BuildOutputDir%%Configuration%\pybridge.pdb" "%__currentScriptDir%src\python\nimbusml\internal\libs\"
)

call "%PythonExe%" -m pip install --upgrade "wheel>=0.31.0"
cd "%__currentScriptDir%src\python"
call "%PythonExe%" setup.py bdist_wheel --python-tag %PythonTag% --plat-name win_amd64
cd "%__currentScriptDir%"

set WheelFile=nimbusml-%ProductVersion%-%PythonTag%-none-win_amd64.whl
if not exist "%__currentScriptDir%src\python\dist\%WheelFile%" (
    echo setup.py did not produce expected %WheelFile%
    goto :Exit_Error
)

md "%__currentScriptDir%target"
copy "%__currentScriptDir%src\python\dist\%WheelFile%" "%__currentScriptDir%target\%WheelFile%"
echo Python package successfully created: %__currentScriptDir%target\%WheelFile%

if "%InstallPythonPackages%" == "True" (
    echo ""
    echo "#################################"
    echo "Installing python packages ... "
    echo "#################################"
    call "%PythonExe%" -m pip install --upgrade nose pytest graphviz imageio pytest-cov "jupyter_client>=4.4.0" "nbconvert>=4.2.0"
    if %PythonVersion% == 2.7 ( call "%PythonExe%" -m pip install --upgrade pyzmq )
    :: Run azureml-dataprep tests only in pyhon 3.7 as its an optional dependency
    if %PythonVersion% == 3.7 ( call "%PythonExe%" -m pip install --upgrade azureml-dataprep )
    call "%PythonExe%" -m pip install --upgrade "%__currentScriptDir%target\%WheelFile%"
    call "%PythonExe%" -m pip install "scikit-learn==0.19.2"
)

if "%RunTests%" == "False" ( 
    goto :Exit_Success
)


echo ""
echo "#################################"
echo "Running tests ... "
echo "#################################"
set PackagePath=%PythonRoot%\Lib\site-packages\nimbusml
set TestsPath1=%PackagePath%\tests
set TestsPath2=%__currentScriptDir%src\python\tests
set TestsPath3=%__currentScriptDir%src\python\tests_extended
set ReportPath=%__currentScriptDir%build\TestCoverageReport
call "%PythonExe%" -m pytest --verbose --maxfail=1000 --capture=sys "%TestsPath1%" --cov="%PackagePath%" --cov-report term-missing --cov-report html:"%ReportPath%"
if errorlevel 1 (
    goto :Exit_Error
)
call "%PythonExe%" -m pytest --verbose --maxfail=1000 --capture=sys "%TestsPath2%" --cov="%PackagePath%" --cov-report term-missing --cov-report html:"%ReportPath%"
if errorlevel 1 (
    goto :Exit_Error
)

if "%RunExtendedTests%" == "True" (
    call "%PythonExe%" -m pytest --verbose --maxfail=1000 --capture=sys "%TestsPath3%" --cov="%PackagePath%" --cov-report term-missing --cov-report html:"%ReportPath%"
    if errorlevel 1 (
        goto :Exit_Error
    )
)

:Exit_Success
endlocal
exit /b %ERRORLEVEL%

:Exit_Error
endlocal
echo Failed with error %ERRORLEVEL%
exit /b %ERRORLEVEL%