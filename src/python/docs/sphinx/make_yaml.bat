@ECHO ON
set PY=%~dp0..\..\..\..\dependencies\Python3.6\python.exe
set PYS=%~dp0..\..\..\..\dependencies\Python3.6\Scripts
set PYTHONPATH=%~dp0..\..\..\..\python
%PYS%\pip install sphinx==1.5.5
%PYS%\pip install sphinx-docfx-yaml
%PYS%\pip install sphinx_rtd_theme
%PYS%\sphinx-build -c ci_script . _build

if exist _build del /Q _build

mkdir _build\ms_doc_ref\
xcopy /S /I /Q /Y /F _build\docfx_yaml\* _build\ms_doc_ref\nimbusml\docs-ref-autogen
del _build\ms_doc_ref\nimbusml\docs-ref-autogen\toc.yml

%PYS%\pip install sphinx==1.6.2
CALL make_md.bat

copy /Y toc.yml _build\ms_doc_ref\nimbusml\toc.yml
xcopy /Y /S _build\md\* _build\ms_doc_ref\nimbusml

del _build\ms_doc_ref\nimbusml\doc-warnings-rx.log
del _build\ms_doc_ref\nimbusml\doc-warnings-rx-all.log
del _build\ms_doc_ref\nimbusml\tutorial.md

echo updating yml...
%PY% ci_script\gen_toc_yml.py -input _build\ms_doc_ref\nimbusml\index.md -temp _build\ms_doc_ref\nimbusml\toc_ref.yml -output _build\ms_doc_ref\nimbusml\toc.yml

echo updating reference links....
%PY% ci_script\update_all_toc_yml.py

echo updating ms-scikit.md to modules.md
del _build\ms_doc_ref\nimbusml\ms-scikit.md
mv _build\ms_doc_ref\nimbusml\modules.md _build\ms_doc_ref\nimbusml\ms-scikit.md