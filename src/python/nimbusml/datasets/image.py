# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
"""
Data about images.
"""
import os


def get_RevolutionAnalyticslogo():
    """
    Return a path to *RevolutionAnalyticslogo.png*.

    .. image:: images/RevolutionAnalyticslogo.png
    """
    this = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(this, "images", "RevolutionAnalyticslogo.png")


def get_Microsoftlogo():
    """
    Return a path to *Microsoftlogo.png*.

    .. image:: images/Microsoftlogo.png
    """
    this = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(this, "images", "Microsoftlogo.png")
