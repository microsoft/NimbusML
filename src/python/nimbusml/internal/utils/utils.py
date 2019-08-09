# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
"""
general utility functions
"""

import logging
import os
import pkg_resources
import tempfile
from datetime import datetime

import six
from pandas import Series

logger_trace = logging.getLogger("nimbusml")


def run_tests(args=None, plugins=None):
    """
    Perform in-process test run using pytest.  See help(pytest.main) for
    details about arguments.
    """
    if args is None:
        my_path = os.path.realpath(__file__)
        my_dir = os.path.dirname(my_path)
        args = ["-x", os.path.dirname(my_dir)]

    import pytest
    pytest.main(args, plugins)


def dir1(obj):
    """
    get the attributes of an object that are not interpreter-related
    """
    return [{a: getattr(obj, a)} for a in dir(obj) if not a.startswith('__')]


def tempdir():
    """
    get the path of the per-session temporary directory
    """
    return tempfile.gettempdir()


def tempfile_(pattern="file", tmpdir=tempdir(), fileext=""):
    """
    character strings which can be used as names for temporary files.
    """
    temp_file_obj = tempfile.TemporaryFile(
        suffix=fileext, prefix=pattern, dir=tmpdir)
    temp_file = temp_file_obj.name
    temp_file_obj.close()  # delete the file already created
    return temp_file


def unlist(mylist):
    """flatten (an irregular) list of lists
    """
    for elem in mylist:
        if isinstance(elem, list):
            for e in unlist(elem):
                yield e
        else:
            yield elem


def try_set(
        obj,
        none_acceptable=False,
        is_of_type=None,
        valid_range=None,
        values=None,
        field_names=None,
        not_implemented=False,
        is_column=False):
    """
    check parameter type, range, options, and etc.
    """
    if not_implemented:
        if obj is not None:
            raise ValueError("Parameter is not currently supported.")
        else:
            return

    if not none_acceptable and obj is None:
        raise ValueError(
            "'None' parameter passed when parameter cannot be none.")

    if is_column:

        def transform_column(col):
            if isinstance(col, (str, six.text_type)):
                return col
            elif isinstance(col, tuple):
                return '.'.join(col)
            elif isinstance(col, list):
                return [transform_column(c) for c in col]
            elif isinstance(col, dict):
                if 'Name' not in col or 'Source' not in col:
                    raise ValueError("invalid column name {0}".format(obj))
                return dict(Name=transform_column(col['Name']),
                            Source=transform_column(col['Source']))
            else:
                msg = "invalid type passed to function {0} in {1})"
                raise ValueError(msg.format(type(col), obj))

        if isinstance(obj, (dict, tuple, list)):
            # nimbusml does not allow tuple (comes from pandas MultiIndex).
            return transform_column(obj)
        elif isinstance(obj, (str, six.text_type)):
            return obj
        else:
            msg = "invalid type passed to function {0} != {1} (expected)"
            raise ValueError(msg.format(type(obj), is_of_type))
    elif is_of_type is not None and obj is not None and \
            not isinstance(obj, is_of_type):
        if is_of_type == str and not isinstance(obj, six.string_types):
            msg = "invalid type passed to function {0} != {1} (expected)"
            raise ValueError(msg.format(type(obj), is_of_type))

    if valid_range is not None:
        for key in valid_range:
            bound = valid_range[key]
            if key == 'Sup' and bound != "Infinity" and obj >= bound:
                raise ValueError("parameter passed >= sup.")
            elif key == 'Inf' and bound != "-Infinity" and obj <= bound:
                raise ValueError("parameter passed <= inf.")
            elif key == 'Max' and bound != "Infinity" and obj > bound:
                raise ValueError("parameter passed > max.")
            elif key == 'Min' and bound != "Infinity" and obj < bound:
                raise ValueError("parameter passed < min.")

    if values is not None and obj not in values:
        raise ValueError("parameter passed not in values.")

    if field_names is not None and not all(
            [key in field_names for key in obj]):
        raise ValueError("parameter passed not in field_names.")

    return obj


def upper_first(string):
    return string[0].upper() + string[1:]


if six.PY2:
    import decorator

    @decorator.decorator
    def trace(func, *args, **kwargs):
        """
        Decorator for tracing enter and exit times
        """

        verbose = 0
        if 'verbose' in kwargs:
            verbose = kwargs['verbose']
        if not isinstance(verbose, six.integer_types):
            raise TypeError(
                "Misaligned parameters. verbose must be int "
                "not '{0}': {1}".format(
                    type(verbose), verbose))
        if verbose > 0:
            logger_trace.info(
                "[%s] enter %s.%s " %
                (datetime.now(),
                 func.__module__,
                 getattr(
                     func,
                     '__qualname__',
                     func.__name__)))
        params = func(*args, **kwargs)
        if verbose > 0:
            logger_trace.info(
                "[%s] exit %s.%s " %
                (datetime.now(),
                 func.__module__,
                 getattr(
                     func,
                     '__qualname__',
                     func.__name__)))

        return params

else:
    from functools import wraps

    def trace(func):
        """
        Decorator for tracing enter and exit times
        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            """
            wrapped function call
            """
            verbose = 0
            if 'verbose' in kwargs:
                verbose = kwargs['verbose']
            if not isinstance(verbose, six.integer_types):
                raise TypeError(
                    "Misaligned parameters. verbose must be int "
                    "not '{0}': {1}".format(
                        type(verbose), verbose))
            if verbose > 0:
                logger_trace.info(
                    "[%s] enter %s.%s " %
                    (datetime.now(),
                     func.__module__,
                     getattr(
                         func,
                         '__qualname__',
                         func.__name__)))
            params = func(*args, **kwargs)
            if verbose > 0:
                logger_trace.info(
                    "[%s] exit %s.%s " %
                    (datetime.now(),
                     func.__module__,
                     getattr(
                         func,
                         '__qualname__',
                         func.__name__)))

            return params

        return wrapper


def compare_shape(pred, X):
    predshapeexists = hasattr(pred, 'input_shape_')
    xshapeexists = hasattr(
        X,
        'shape') or isinstance(
        X,
        Series) or isinstance(
        X,
        list)

    # Should we raise an error in this case?
    if (predshapeexists and not xshapeexists) or (
            not predshapeexists and xshapeexists):
        raise ValueError(
            'Reshape your data. Shape of input is different from '
            'what was seen in `fit`')

    if predshapeexists and xshapeexists:
        predcolcount = pred.input_shape_[1]
        xcolcount = 1
        if isinstance(X, list) and isinstance(X[0], list):
            xcolcount = len(X[0])
        elif hasattr(X, 'shape') and len(X.shape) == 2:
            xcolcount = X.shape[1]

        if predcolcount != xcolcount:
            raise ValueError(
                'Reshape your data. Shape of input is different from what '
                'was seen in `fit`')


def set_shape(pred, X):
    if isinstance(X, Series):
        pred.input_shape_ = (X.shape[0], 1)
    elif hasattr(X, 'shape'):
        if len(X.shape) == 1:
            pred.input_shape_ = (X.shape[0], 1)
        else:
            pred.input_shape_ = X.shape
    elif isinstance(X, list):
        if isinstance(X[0], list):
            pred.input_shape_ = (len(X), len(X[0]))
        else:
            pred.input_shape_ = (len(X), 1)

def set_clr_environment_vars():
    """
    Set system environment variables required by the .NET CLR.
    Python 3.x only, as dotnetcore2 is not available for Python 2.x.
    """
    if six.PY2:
        pass
    else:
        from dotnetcore2 import runtime as clr_runtime
        dependencies_path = None
        try: 
            # try to resolve dependencies, specifically libunwind for Linux
            dependencies_path = clr_runtime.ensure_dependencies()
        except:
            pass
        # Without this, Linux versions would require the ICU package
        os.environ['DOTNET_SYSTEM_GLOBALIZATION_INVARIANT'] = 'true'
        # Will be None for Windows
        if dependencies_path is not None:
            os.environ['LD_LIBRARY_PATH'] = dependencies_path

def get_clr_path():
    """
    Return path to .NET CLR binaries.
    Use dotnetcore2 package if Python 3.x, otherwise look for libs bundled with
    NimbusML.
    """
    if six.PY2:
        return get_mlnet_path()
    else:
        from dotnetcore2 import runtime as clr_runtime
        libs_root = os.path.join(clr_runtime._get_bin_folder(), 'shared', 
                                'Microsoft.NETCore.App')

        # Search all libs folders to find which one contains the .NET CLR libs
        libs_folders = os.listdir(libs_root)
        if len(libs_folders) == 0:
            raise ImportError("Trouble importing dotnetcore2: "
                                "{} had no libs folders.".format(libs_root))
        clr_path = None
        for folder in libs_folders:
            if os.path.exists(os.path.join(libs_root, folder, 
                                           'Microsoft.CSharp.dll')):
                clr_path = os.path.join(libs_root, folder)
                break
        if not clr_path:
            raise ImportError(
                "Trouble importing dotnetcore2: Microsoft.CSharp.dll was not "
                "found in {}.".format(libs_root))
        return clr_path

def get_dprep_path():
    """
    Return path to DataPrep binaries if its installed, None otherwise
    """
    try:
        from azureml.dataprep.api.engineapi.engine import _get_engine_path
        return os.path.dirname(_get_engine_path())
    except ImportError:
        pass
    return ''

def get_mlnet_path():
    """
    Return path to ML.NET binaries.
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 
                                        'libs'))
