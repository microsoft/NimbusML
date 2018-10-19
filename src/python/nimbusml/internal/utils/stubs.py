# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
from functools import wraps


def telemetry_pause():
    pass


def telemetry_unpause():
    pass


def telemetry_capture_call(name=None):
    pass


def telemetry_transform(func):
    """
    Decorator for sending telemetry for nimbusml data transforms.
    """

    @wraps(func)
    def wrapper(*args, **kargs):
        """
        wrapped function call
        """
        params = func(*args, **kargs)

        # telemetry_info = func.__name__
        # nimbusml_bridge(telemetry_info=telemetry_info,
        #            capture_telemetry_only=True)

        return params

    return wrapper
