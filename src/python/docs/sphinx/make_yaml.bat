@if not defined _echo @echo off
if exist %~dp0build (rmdir /S /Q %~dp0build)
if exist %~dp0..\..\..\..\dependencies\Python3.7 (
echo "Python3.7 exists"
) else (
echo "Please run build.cmd under NimbusML with Python3.7's configuration first"
call exit /b
)
echo "###Downloading Dependencies######"
echo "Downloading Dependencies "
set PY=%~dp0..\..\..\..\dependencies\Python3.7\python.exe
set PYS=%~dp0..\..\..\..\dependencies\Python3.7\Scripts
set PYTHONPATH=%~dp0..\..\..\..\python
echo "Installing sphinx-docfx-yaml "
call %PY% -m pip -q install sphinx-docfx-yaml
echo "Installing sphinx "
call %PY% -m pip -q install sphinx==2.1.1
echo "Installing sphinx_rtd_theme "
call %PY% -m pip -q install sphinx_rtd_theme
echo "Installing NimbusML "
call %PY% -m pip -q install nimbusml
echo "#################################"
echo.

echo.
echo "#################################"
echo "Running sphinx-build "
echo "#################################"
call %PY% -m sphinx -c %~dp0ci_script %~dp0 %~dp0_build

echo.
echo "#################################"
echo "Copying files "
echo "#################################"
call mkdir %~dp0_build\ms_doc_ref\
call xcopy /S /I /Q /Y /F %~dp0_build\docfx_yaml\* %~dp0_build\ms_doc_ref\nimbusml\docs-ref-autogen

echo.
echo "#################################"
echo "Running make_md.bat"
echo "Fixing API guide
echo "#################################"
:: Todo: //Have a bug here stop iterator
call make md
call %py% %~dp0ci_script\fix_apiguide.py

call copy /Y %~dp0toc.yml %~dp0_build\ms_doc_ref\nimbusml\toc.yml
call xcopy /Y /S %~dp0_build\md\* %~dp0_build\ms_doc_ref\nimbusml
:: Append the text in index.md under tutorial.md

echo.
echo "#################################"
echo "updating yml......."
echo "#################################"
call %PY% %~dp0ci_script\gen_toc_yml.py -input %~dp0_build\ms_doc_ref\nimbusml\index.md -temp %~dp0_build\ms_doc_ref\nimbusml\toc_ref.yml -output %~dp0_build\ms_doc_ref\nimbusml\toc.yml

echo.
echo "#################################"
echo "updating reference links...."
echo "#################################"
call %PY% %~dp0ci_script\update_all_toc_yml.py

echo.
echo "#################################"
echo "updating ms-scikit.md to modules.md"
echo "#################################"
call move %~dp0_build\ms_doc_ref\nimbusml\modules.md %~dp0_build\ms_doc_ref\nimbusml\ms-scikit.md

echo.
echo "#################################"
echo "Cleaning files"
echo "#################################"
call mkdir %~dp0build
call move %~dp0_build\ms_doc_ref %~dp0\build\
call more +29 %~dp0build\ms_doc_ref\nimbusml\index.md >> %~dp0build\ms_doc_ref\nimbusml\overview.md
call del /Q %~dp0build\ms_doc_ref\nimbusml\*log
call del /Q %~dp0build\ms_doc_ref\nimbusml\concepts.md
call del /Q %~dp0build\ms_doc_ref\nimbusml\index.md
call del /Q %~dp0build\ms_doc_ref\nimbusml\toc.yml
:: call rmdir /S /Q %~dp0_build

echo.
echo "#################################"
echo "#########Built Finished##########"
echo "#################################"