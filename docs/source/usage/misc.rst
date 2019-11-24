Misc
=======

Systematics configuration
-------------------------

Systematic uncertainties are a key ingredient in any data analysis and typically imply the strongest headaches when adding
them. Since they are used in several places throughout the analysis code, e.g. as overlaid histogram in plots, as input
to the limit setting etc, a common interface is implemented which can be configured (surprise!) by a yaml file. The code
distinguishes three different main types of systematics:

1.) Normalisation uncertainties:

These are uncertainties which only change the weight of a specific event. This category are typically all scale factors
(e.g. electron/muon ID, trigger etc.). When running ELBrain these will be stored in the output tree in the 'Nominal'
directory as the *relative* difference to the nominal weight, i.e. weight_unc = systematic weight / nominal weight. This
is also the baseline assumption when these uncertainties are handled internally. This affects one class of uncertainties
which share in fact the same nature but are treated very differently, namely theory uncertainties stored as weights (e.g.
the renormalisation and factorisation scale uncertainties in Sherpa). These fall under category 3.

2.) Shape uncertainties:

These are uncertainties which shift the kinematics of a certain object and thus have a shape impact on an object/event
level. Typical examples are energy scale/resolution uncertainties. These are stored in dedicated trees in different
directories in the ntuples with the directory name being the name of the uncertainty. To evaluate the impact the distribution
will be fetched from these trees and compared to the nominal distribution.

3.) Custom uncertainties:

This class takes care of anything which does not fall in either of the previous categories and typically involves custom
code to be executed as already discussed for the theory uncertainties.

The different types can be set via configurable options of a common *Systematics* class.

.. code-block:: python

    PRW_DATASF:
      title: 'PRW syst'
      type: 'scale'
      variation: "updown"
      group: "PRW"

This will create a systematic uncertainty named PRW_DATASF which is a *scale* uncertainty being part of the *group* PRW
uncertainties providing both an up and down *variation*.
When processing this uncertainty the code will look for two branches in each MC input file: Nominal/tree_name.PRW_DATASF__1down
and Nominal/tree_name.PRW_DATASF__1up, make a histogram based on the nominal setting but with a modified
weight being nominal_weight * PRW_DATASF__1down/up which will be the systematically varied histogram for this uncertainty.
The title attribute is used for tables/plots e.g. within TRexFitter

A custom uncertainty is very similar, but has additional arguments. The type must be custom in order to invoke the custom
handling. Via the call attribute a function call can be defined which will be executed within this code. In the example
below a new instance of the *TheoryUncertaintyProvider* class will be created and the *get_envelop()* function will be
called. If one adds additional functions/classes in the underlying python code one needs to guarantee that the corresponding
module is imported in the *SystematicsAnalyser* module. Additionally one can (this will work for any systematic) define
a list of samples which are affected by this particular uncertainty (*samples* attribute). The *variation* in this case
is *custom* which means that the code decides how up/down are defined. As for the normalisation and shape analysis this
can also be *updown*, *up* or *down*.

.. code-block:: python

    theory_envelop:
      variation: "custom"
      type: custom
      call: "TheoryUncertaintyProvider().get_envelop(self, dumped_hist_path)"
      samples:
        - Zjets
        - Others


Plot options
------------

Since there are many different (mainly style) options which can be set for each plot config they are summarised here. If
no value of any of the attributes is provided a default value will be used. Within the *PyAnalysisTools* package the
"default" default configuration is provided in (PlottingUtils/plot_config_defaults.yml). However, since this might not
meet the needs or taste of the user it can be overwritten by putting a config file with the same name into any directory
within the user's analysis package (e.g. ELYourFancyAnalysis/macros/plot_config_defaults.yml). If such a file is found
the defaults will be taken from there. Of course the user only needs to specify the attributes he/she likes to overwrite
while if any attribute is not provided by neither the users default and plot config the package's default config is used.

* enable_range_arrows (bool): enable/disable arrows in ratio if ratio is out of range


Check ntuple production completeness
------------------------------------

Typically you run your ntuple production on the grid and want to make sure that all your dataset have been processed
completely. An easy check is to compare the number of processed events to the number of generated events and in case
some jobs failed resubmit them. This can be done with the *analyse_ntuples.py* tool which is shipped as an executable.
It reads in your dataset list (the one you used for submission), checks your downloaded ntuples against it and compares
the number of processed events against AMI (Note: you need to load the pyAMI library to get the client). If a dataset
is missing, i.e. not processed at all, or if it is incomplete, i.e. you have processed not all events e.g. due to
failing grid jobs, it will be reported and a new dataset list will be rewritten containing only the failed datasets. In
the new dataset list you will find two tags, *missing* and *incomplete*. Incomplete dataset you will want to resubmit
with the same version tag to re-run only the failed jobs, while the missing ones you might want to resubmit as a new
version due to *broken* states etc.

.. code-block:: python

     analyse_ntuples.py /storage/hepgrp/morgens/LQ/ntuples/v29/ -ds run/configs/leptoquarks/datasets.yml -r


Positional arguments:

* input_path : path containing downloaded ntuples

Optional arguments:

* dataset_list: file containing datasets submitted for ntuple production
* resubmit: boolean argument to enable writing of resubmit dataset list file - stored in the same directory as the input with the same name, but *_resubmit* suffix
* filter: pattern used to ignore keys in input dataset file (stored as dictionary)
