"""
Microsoft Machine Learning for Python
"""

__version__ = '1.3.0'

# CoreCLR version of MicrosoftML is built on Windows.
# But file permissions are not preserved when it's copied to Linux.
# The pyridge.so library has to be executable to use nimbusml on Linux
# so here is a workaround to achieve that.
import os
import sys
# Silence Cython warnings that check for different binary versions of numpy
# across each thirdparty module that uses numpy. It is standard practice to
# silence this warning as seen in numpy/__init__.py:
# https://github.com/numpy/numpy/pull/432/commits/170ed4e33d6196d724dc18ddcd42311c291b4587?diff=split
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")

from .internal.utils.data_roles import Role
from .internal.utils.data_schema import DataSchema
from .internal.utils.data_stream import BinaryDataStream
from .internal.utils.data_stream import DprepDataStream
from .internal.utils.data_stream import FileDataStream
from .internal.utils.utils import run_tests
from .pipeline import Pipeline

if sys.platform.lower() == "linux":
    pkg_path = os.path.dirname(os.path.realpath(__file__))
    dotso = os.path.join(pkg_path, "internal", "libs", "pybridge.so")
    mode = oct(os.stat(dotso).st_mode & 0o777)
    if mode != "0o755":
        os.chmod(dotso, 0o755)

# clean up the package namespace
del os, sys

__all__ = [
    'Pipeline',
    'DataSchema',
    'FileDataStream',
    'BinaryDataStream',
    'Role'
]
