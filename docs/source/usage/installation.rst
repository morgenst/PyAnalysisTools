Installation
============

To use PyAnalysisTools locally clone the repository (ideally via kerberos authentication)

.. code-block:: python

    git clone https://:@gitlab.cern.ch:8443/ELBrain/PyAnalysisTools.git

Next install the dependencies via pip:

.. code-block:: python

    cd PyAnalysisTools
    pip install --user -r requirements.txt


.. note::

    IMPORTANT: The installation of all dependencies is ONLY needed if you want to run it standalone. If you work on
    lxplus or any other system which has access to cvmfs you can load the majority of the dependencies via cvmfs and
    only need to install those which are not available. To do so you can either first load all modules from cvmfs
    (typically with one of your analysis setup scripts) and the call the pip install above or use the following command.

    .. code-block:: python

        pip install --user oyaml pathos future
