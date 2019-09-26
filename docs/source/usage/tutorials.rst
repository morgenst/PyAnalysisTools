Tutorials
=========

In the following you will find several tutorials explaining the basic usage.

Making a plot
-------------

Creating plots is one of the core functionality and the code is designed such that everything can be configured via
several configuration files defining the content. All configuration files are written in **yaml** To understand the concept behind it one needs to understand the meaning
of these config files. To make a *standard* plot one typically needs at least three configuration files:

1.) **The dataset config**

The dataset config maps a certain dataset to its meta-data. In an usual analysis you would map a dataset ID (*dsid*) for
a given MC sample to a set of information such as its *cross section*, *human readable name*, *generator* etc. All this
information is encapsulated in a **Dataset** class and a dictionary linking the dsids to this information is stored in
a config file. The metadata should by default be taken from PMG. For convenience a corresponding script is provided to
automatically generate this config file given a list of datasets.

.. note:: What happens behind the scenes

    Internally the dsid is parsed from the input file name, that's why a certain naming scheme has to be followed (for now),
    and a process is assigned to it (see below). During the processing all information needed for e.g. scaling to luminosity,
    is taken from the parsed config file.


2.) **The process config**

Typically you will have many samples you want to analyse and eventually combine, e.g. you might want to combine all
binned *Z+jets* Sherpa samples into a single component shown in your plots as the *Z+jets* component. Also you want to
assign a certain label presented in table headers or legends, treat it differently depending on whether it is a signal,
background or data sample etc. This is what the process config is supposed to handle. As in many other cases the parsed
information is encapsulated in a dedicated class called Process.
A usual process in the configuration file looks like this:

.. code-block:: python

    Zjets:
      Type: Background
      Label: Z+jets (Sherpa)
      Color: kRed+2
      Draw: Hist
      Style: 1001
      SubProcesses: [re.Zmumu.*, re.Zee.*, re.Ztautau.*, re.Znunu.*]

This defines a process named *Zjets* which is build out of any sample which name matches any of the regular expressions
defined in the *SubProcesses* argument (see above for more details on the sample names). If drawn it's drawn as a
histogram (*Draw* option) with a *FillStyle* of 1001 (*Style* option) using the color *kRed+2*. The assigned label used
in legends or table headers is defined via the *Label* argument.

3.) **The plot config**

The plot config defines the content and outline and is implemented in the PlotConfig.py class. Over time the possible
arguments grew quite significantly, but just a few are needed while the majority is for formatting purpose. Many examples
from an actual analysis can be found here: **LINK TO LQ**

A simple plot config looks like this:

.. code-block:: python

   lq_mass_max:
    dist : lq_mass_max / 1000.
    xtitle : "m_{l,jet}^{max} [GeV]"
    bins : 20
    xmin : 50.
    xmax : 2500.
    logy: True
    logx: True

This represents a dictionary in yaml with the name of the histogram being the key, i.e. *lq_mass_max*. The content is
defined in the *dist* argument which can contain any valid TFormula. The given string will be used to fill the histogram
from the input tree (calling TTree::Project). The histogram definition, i.e. the number of bins as well as the x-axis range
are defined by the respective arguments (*bins*, *xmin*, *xmax*). The other arguments are for formatting purpose and
hopefully are self-explaining.

.. note::

   The arguments change for 2D plots since one needs to explicitly specify the x- and y-axis binning, i.e *bins* is
   replaced by *xbins* and *ybins*

There are several more options which can be configured and the most commonly used ones are summarised below:

.. code-block:: python
    :linenos:

    watermark: A watermark label added to each canvas after ATLAS, e.g. Internal, Preliminary
    luminosity: float or dictionary defining the luminosity

.. note::

    ADD A COMMENT on lumi and float/dict

Now since each plot configuration is distinct it would be very cumbersome to define each common arguments, e.g. the
watermark, luminosity etc again and again and again. Thus, for convenience, there is the possibility to define a common
section in the config file defining all shared properties named *common*. This is treated a special plot config which is
parsed and propagated to each plot config not matching the name *common*

.. code-block:: python

    common:
      Watermark: Internal
      Lumi:
        mc16a: 36.24
        mc16d: 44.3
        mc16e: 58.45
      yTitle: Events
      outline: stack
      ratio: True
      grid: True
      weight: weight
      merge: True
      decor_text_y: 0.75
      lumi_text_y_ratio: 0.82
      merge_mc_campaigns: True
      ordering:
        - ttbar
        - Zjets
        - Others
        - Data
      ratio_config:
        draw: Marker
        ymin: 0.5
        ymax: 1.5
        ytitle: Data / SM


.. note::

    In many cases one will pass several plot configuration files to the steering script which will be merged internally.
    In case they have different common sections they will be compared and an interactive session will start in which the
    user will be asked which of settings should be applied. Currently it is not possible to propagate a common config
    separately to each plot config defined in the corresponding file (this would usually make not much sense as one should
    start two instances of the plotting code).

For ratio plots (needs proper phrasing): If one wants to show the uncertainties of points which are outside of the range
one needs to set draw_option: "e0" in the ratio_config.

Setting limits
--------------

The currently implemented limit setting code is based on top of `TRExFitter <https://gitlab.cern.ch/TRExStats/TRExFitter>`_. An interface to pyhf (**ADD LINK**) is
currently worked on. The inputs are expected to be histograms for both the nominal selection as well as for each systematics
uncertainty.