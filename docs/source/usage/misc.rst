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
* cuts: selection applied when fetching histogram from tree (default: None, i.e. no cuts applied)
* add_text: add additional text to canvas
* weight: specify which weights to apply when fetching histogram - any expression that works in TTree::Draw
* process_weight: same as weight but only for a specific process (default: None, i.e. nothing applied)
* dist: name of the distribution to plot, e.g. muon_pt / 1000. plots muon p_T in GeV
* style: ROOT draw style, i.e. line, marker, fill
* rebin: rebin histogram - symmetric (providing single factor) or asymmetric (providing list of bin boundaries)
* ignore_rebin: ignore rebinning settings
* ratio: enable ratio pad (data/MC)
* blind: selection for blinding distribution, e.g. above a certain mass threshold or whatever defines your signal region
* ordering: list of processes in order of drawing first to last
* [xyz]bins: number of [xyz] bins
* [xyz]min[_log]: [xyz]-axis minimum [if log-scale]
* [xyz]title: [xyz]-axis title
* [xyz]title_offset: [xyz]-axis title offset
* [xyz]title_size: [xyz]-axis title size
* axis-labels: list of bin labels (default applied to x-axis)
* labels: legend labels - typically parsed from process config
* color: set of colors applied to histograms/stacks etc
* draw_option: explicit draw option (overwrites `draw`)
* normalise_range: range in which normalisation should be calculated (default: entire range, i.e. -1, -1)
* ratio_config: plot configuration for ratio plot (all non explicitly specified settings are taken from provided plot config)
* signal_scale: optional scaling factor to applied to all signal samples, e.g. to make tiny signals visible
* lumi: luminosity used for scaling MC and printed to canvas - takes either float or dictionary of lumi per MC campaign (to disable lumi text in canvas set lumi to -1)
* lumi_text:
* total_lumi: total luminosity (typically calculated from lumi dictionary)
* watermark: watermark after `ATLAS` - typically Internal/Preliminary
* stat_box: enable statistics box - if multiple objects are drawn stat boxes are drawn with matching colors
* normalise: normalise plot depending on normalise_range (default: unit area)
* no_data: do not draw data
* ignore_style: ignore all style settings
* calcsig: enable significance calculation (applying cut along x-axis to signal and background) and draw like ratio
* enable_legend: enable adding legend (default: True)
* make_plot_book: produce plot book, i.e. all plots will be stored in a single pdf
* grid: enable drawing of grid (default: False)
* log[xyz]: Make xyz-axis log scale
* decor_text: additional text to add to canvas
* disable_bin_merge: disable merging of under- and overflow bins (default: False)
* enable_range_arrows: turn on drawing arrows in ratio hist if ratio is out of range
* merge: (default: True)
* signal_extraction: extract signal from all histograms - needed to treat them differently, e.g. overlay as hist on stack
* merge_mc_campaigns: enable merging of MC production campaigns (default: True)
* name: name of the plot config - will be used as file name for pdf
* draw: definition of draw style, e.g. hist, marker, line
* outline: outline of the plot, e.g. hist, stack
* legend_options: optional legend options, e.g. position, number of columns
* lumi_precision: digits of lumi precision (default: 1)
* yscale[_log]: multiplicative off-set of y-axis maximum (_log if log-scale) if not explicitly specified ymax (default: 1.2 [100.])
* norm_scale: value to which will be normalised to (default: 1.0)
* lumi: luminosity used as label and internal calculation; can be single value or dictionary with lumi for each MC campaign
* watermark_size[_ratio]: text size of watermark [if ratio is added to canvas]
* watermark_offset[_ratio]: text offset of watermark w.r.t `ATLAS` [if ratio is added to canvas]
* watermark_[xy][_ratio]: x,y position of watermark text [if ratio is added to canvas]
* decor_size[_ratio]: size of decoration text [if ratio is added to canvas]
* decor_[xy][_ratio]: x,y position of decoration text [if ratio is added to canvas]
* lumi_size[_ratio]: size of lumi text [if ratio is added to canvas]
* lumi_[xy][_ratio]: x,y position of lumi text [if ratio is added to canvas]


Merging ntuples
---------------

When doing your analysis you will likely submit production jobs for your ntuples to the grid. After downloading all
processed samples you will end up with a bunch of directories with long (and perhaps cryptic) names and several tentatively
small root files in each directory. Since this kind of output is rather inconvenient for further post-processing, e.g. plotting,
you likely want to merge the root files into single files - one per sample. This job is done by the *merge_ntuple.py*
script. To run it you just need to do this:

.. code-block:: python

     merge_ntuples.py INPUT_PATH -o OUTPUT_PATH --filter r10724 --tag MC16e

The *INPUT_PATH* and *OUTPUT_PATH* are required arguments pointing to the path containing the downloaded datasets and
the destination directory to store the merged ntuples. *filter* and *tag* are optional arguments which come in handy if
you have to deal with different MC campaigns etc. *filter* will only considered datasets whose name matches the filter
arguments (in the example using the reco-tag for MC16e production) and *tag* will add an additional suffix to the output
file name, i.e. the example above will create files named *ntuple-DSID.MC16e.root*.
If for some reason you have to deal with recorded data datasets not provided as period containers, but on a per run basis,
and you'd like to still have period container merged ntuples you can pass a configuration file (*data_summary*)
providing the run number to period mapping.
Other optional arguments are

* merge_dir: temporary directory to perform hadd (adding of ntuples). Needed if destination file system does not allow overwrite operations
* ncpu: number of parallel merge jobs to execute
* force: run hadd -f, i.e. force creation



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


Applying selections
-------------------

Often you want to apply more restrictive object and event level selections after the ntuples are produced. The RegionBuilder
provides a convenient interface used throughout the entire code base. The concept is trivial, given a selection configuration
file a *set of cuts* is created and assigned to a *region*.
The config may look like this:

.. code-block:: python

     RegionBuilder:
      auto_generate: False
      common_selection:
        event_cuts:
          - "1 ::: Preselection"
          - "muon_n == 2 ::: Two muons"
          - "jet_n > 0 ::: At least 1 jet"
          - "muon_pt[0] / 1000. > 65. ::: muon \\pT{} > 65~\\GeV{}"
          - "inv_mass_muons > 400. ::: \\minv{} > 400~\\GeV{}"
          - "ht_jets + ht_leptons > 350. ::: \\HT{} > 350~\\GeV{}"

        post_sel_cuts:
          - "lq_mass_max / 1000 > 400. ::: \\mLQmax{} > 400~\\GeV{}"


      regions:
        SR_mu_one_btag:
          n_lep: 2
          n_electron: 0
          n_muon: 2
          disable_taus: True
          same_flavour_only: True
          label: "SR #mu^{#pm}#mu^{#mp} 1 b-tag"
          event_cuts:
            - "Sum$(jet_has_btag) == 1 ::: 1 b-tagged jet"
            - "jet_has_btag[0] == 1 ::: leading jet b-tagged"

        SR_mu_bveto:
          n_lep: 2
          n_electron: 0
          n_muon: 2
          disable_taus: True
          same_flavour_only: True
          label: "SR #mu^{#pm}#mu^{#mp} b-veto"
          event_cuts:
            - "Sum$(jet_has_btag==1) == 0 ::: b-tag veto"

This will define two regions, *SR_mu_one_btag* and *SR_mu_bveto* with a set of common and distinct cut.
The common selection will be applied to each defined region **prior** to the specific selection. This is just for convenience
to avoid repeat the common part. Similarly a common selection applied after the region selection can be defined via the
*post_sel_cuts* configurable. Each of the selection string will be converted to a **Cut** object which accepts any configuration
as this

.. code-block:: python

    "SPECIFIER: CUT ::: NAME"

*SPECIFIER* is a string specifying on which kind of inputs the cut should be applied. This can be one of the following

1.) "DATA": applied only on data

2.) "MC": applied only on MC

3.) "TYPE_PROCESS": applied on a specific process named *PROCESS*

The *CUT* itself must be a string which root can translate. This may include sum handy selections like *Sum$*, *Length$*, etc (`ROOT TTree <https://root.cern.ch/doc/master/classTTree.html#a73450649dc6e54b5b94516c468523e45>`_)
Finally, *NAME* defines a custom string assigned as the cut name which will be printed e.g. in cutflow tables. This must
be separated by the *CUT* via **:::** (3 colons) to allow for ROOT specific calls such as *TMatH::Pi*.
Beside event selection cuts, the RegionBuilder is also able to apply object specific cuts, e.g. require two muons with
:math:`p_{T}` at least 30 GeV and :math:`|\eta|` less than 2.5 can be done as follows:


.. code-block:: python

     RegionBuilder:
      auto_generate: False
      common_selection:
        good_muon:
          - "muon_pt > 30"
          - "abs(muon_eta) < 2.5"

        SR_mu_bveto:
          n_lep: 2
          n_electron: 0
          n_muon: 2


.. note::

    Internally this will be translated to a TCut string checking that the number of leptons matches both the number of
    leptons in the event as well as the number of selected leptons, the cut will look like this:

    "muon_n == 2 && Sum$(muon_pt > 30. && abs(muon_eta)) == muon_n
Since you man not always want to require exactly *N* leptons you can change to operator to *leq* (<=) or *geq* (>=) by
setting the *electron_operator* or *muon_operator*.
The *auto_generate* option let you generically set up regions for any combination of *N* leptons with up to *x* electrons
and *y* muons. (Note: This hasn't been tested recently, so please file a bug report on jira if something is not working
for this option)

There are several other options which can be set for a region mainly for limit setting purpose:

* label: Custom label used for plotting, tables, limits
* norm_region (boolean): define region as normalisation region
* norm_background: define list of samples and normalisation parameters to be constraint in this region
* val_region (boolean): define region as validation region, i.e. not included in fit, but check modeling of best fit values
* channel: name to define a given channel, not the label (used in limit setting only)
* binning: binning of observable used in limit fit. Can be:
    * equidistant binning: min_b1, min_b2, max value (e.g. 300, 500, 8000)
    * asymmetric binning: list of bin borders - supports eval (e.g. eval[300 + i*50 for i in range(5)] + [600 + i*100. for i in range(14)] +[2000, 8000.])
    * optimised binning: using TRexFitter's auto binning (e.g. '"AutoBin","TransfoF",5.,10.')


One example showing the different settings (names should be easy to guess):

.. code-block:: python

    RegionBuilder:
      auto_generate: False
      common_selection:
        event_cuts:
          - "jet_n > 0"
          - "electron_pt[0] / 1000. > 65."
          - "lq_mass_max / 1000 > 300."

      regions:
          SR_el_btag:
              n_lep: 2
              n_electron: 2
              n_muon: 0
              disable_taus: True
              same_flavour_only: True
              label: "SR"
              channel: "e^{#pm}e^{#mp} b-tag"
              event_cuts:
                - "inv_mass_electrons > 400."
                - "jet_has_btag[0] == 1"
                - "Sum$(jet_has_btag) == 1"
                - "ht_leptons + ht_jets > 350."
                - "jet_n > 0"
                - "electron_pt[0] / 1000. > 65."
            TopCR_el:
              norm_region: True
              norm_backgrounds:
               ttbar:
                norm_factor: mu_top
              n_lep: 2
              n_electron: 2
              n_muon: 0
              binning: 300., 8000.
              label: "TopCR"
              disable_taus: True
              same_flavour_only: True
              event_cuts:
                - "jet_n > 1"
                - "Sum$(jet_has_btag) == 2"
                - "inv_mass_electrons > 130."

            VR_el_bveto:
              n_lep: 2
              n_electron: 2
              n_muon: 0
              disable_taus: True
              same_flavour_only: True
              norm_region: False
              val_region: True
              binning: '"AutoBin","TransfoF",5.,10.'
              norm_backgrounds:
               Zjets:
                norm_factor: mu_Z
              label: "VR"
              event_cuts:
                - "inv_mass_electrons < 400."
                - "inv_mass_electrons > 250."
                - "Sum$(jet_has_btag) == 0"
            ZCR_el_bveto:
              n_lep: 2
              n_electron: 2
              n_muon: 0
              norm_region: True
              binning: "eval[300 + i*50 for i in range(5)] + [600 + i*100. for i in range(14)] +[2000, 8000.]"

              norm_backgrounds:
               Zjets:
                norm_factor: mu_Z
              disable_taus: True
              same_flavour_only: True
              label: "ZCR"
              event_cuts:
                - "Sum$(jet_has_btag) == 0"
                - "inv_mass_electrons < 250."
                - "inv_mass_electrons > 130."

