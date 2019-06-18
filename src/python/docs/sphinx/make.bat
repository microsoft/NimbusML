@ECHO OFF

pushd %~dp0
set PYTHONINTERPRETER=%~dp0..\..\..\..\dependencies\Python3.6\python.exe
set SPHINXOPTS=-j 4

REM Command file for Sphinx documentation

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=%PYTHONINTERPRETER% -m sphinx
)
set SOURCEDIR=.
set BUILDDIR=%~dp0_build
set SPHINXPROJ=microsoftml

if "%1" == "" goto html: 
set format=%1
goto next:

:html:
set format=html

:next:

@echo remove %BUILDDIR%\%format%
call rmdir /s /q %BUILDDIR%\doctrees
call rmdir /s /q %BUILDDIR%\%format%
if exist %BUILDDIR%\_static rmdir /S /Q %BUILDDIR%\_static
if exist %BUILDDIR%\%format% goto issue:

%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
	echo.
	echo.The Sphinx module was not found. Make sure you have Sphinx installed,
	echo.then set the SPHINXBUILD environment variable to point to the full
	echo.path of the 'sphinx-build' executable. Alternatively you may add the
	echo.Sphinx directory to PATH.
	echo.
	echo.If you don't have Sphinx installed, grab it from
	echo.http://sphinx-doc.org/
	exit /b 1
)

call %SPHINXBUILD% -M %format% %SOURCEDIR% %BUILDDIR% %SPHINXOPTS%
goto end

:help
call %SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS%
goto end:

:issue:
@echo An issue happened. Check %BUILDDIR%\%format% is not here.

:end
popd
