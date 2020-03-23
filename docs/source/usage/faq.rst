Frequently Asked Questions
===========================

A bunch of questions for which I tend to forget the answers or have written to users many times.

1.) How to debug unit tests?

Testing unit tests in CI might be cumbersome for debugging. Luckily docker helps us again.

.. code-block:: python

    gitlab-runner exec docker unittest --docker-image=pyanapy2 --docker-pull-policy="if-not-present"

Replace `unittest` by whatever test you want to run.

.. note::

    Since this can read the variables of the CI config you have to change .gitlab-ci.yml containing your username and
    password for kinit. REMEMBER to remove this before submitting, otherwise well you know...


2.) I run on a MAC OSX and loading root numpy crashes?

One of the fun things with root/root_numpy and mac (among a few others). If you see a stacktrace like this

.. code-block:: python

     Traceback (most recent call last):
      File "<string>", line 1, in <module>
      File "/Users/../lib/python2.7/site-packages/root_numpy/__init__.py", line 1, in <module>
        from ._tree import (
      File "/Users/../lib/python2.7/site-packages/root_numpy/_tree.py", line 5, in <module>
        from . import _librootnumpy
    ImportError: dlopen(/Users/..lib/python2.7/site-packages/root_numpy/_librootnumpy.so, 2): Library not loaded: @rpath/libCore.so
      Referenced from: /Users/../lib/python2.7/site-packages/root_numpy/_librootnumpy.so
      Reason: image not found

then it's simply a missing link to the root libraries which you can via install_tool:

.. code-block:: python

    install_name_tool -add_rpath $ROOTSYS/lib/ ~/Library/Python/2.7/lib/python/site-packages/root_numpy/_librootnumpy.so


3.) Do I need to install all dependency requirements?

Short: It depends.
Long: Dependencies do only need to be installed if not available. In particular with modules linked to ROOT you may
experience incompatibilities. If you work in an environment which has access to cvmfs you should load all available modules
from there and only install missing ones via pip. The easiest is to setup your environment and once you loaded from cvmfs
run `pip install` to install only the missing ones. Locally or in docker containers you usually have to install all
dependencies.