.. rxtitle:: Installation Guide
.. rxdescription:: Installation guide for users

==================
Installation Guide 
==================

Supported Platforms 
-------------------

Release 0.6:
   * Windows 10, Ubuntu 14.04, Ubuntu 16.04, CentOS 7, RHEL 7, Mac OS 10.11, 10.12, 10.13


Supported Python Version 
------------------------

.. image:: _static/images/supported_version.png # Post processed to md table in build command

Required Packages 
---------------------

The library requires the following dependencies, which will be installed automatically:
*numpy*, *pytz*, *six*, *python-dateutil*, *pandas*, *scikit-learn*, *scipy*.

Installation 
-------------

``nimbusml`` can be installed using ``pip``:

.. code-block:: console

   pip install nimbusml

For a quick test, please run:

.. code-block:: console

    python -m nimbusml.examples.LightGbmClassifier

Building
--------------------

The ``nimbusml`` package can also be built from the `source repo <https://github.com/Microsoft/ML.NET-for-Python>`_
on Github. For more details about building and testing, please refer to our `GitHub repo <https://github.com/Microsoft/ML.NET-for-Python>`_

Contributing
------------

This is an open source package and we welcome contributions. The source code for the  ``nimbusml`` package is `available in GitHub <https://github.com/Microsoft/ML.NET-for-Python>`_.
