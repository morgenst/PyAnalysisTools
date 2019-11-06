Modules
=======

Several modules are provided

Core modules
------------

The base module contains the core functionality used as underlying structure for all analysis tools. It provides basic
interfaces for reading and writing data in different formats (root files, yaml, json), logging, batch job handling and
others.




Plotting tools
--------------

Anything which is related to plotting starting from making histograms towards style tuning are provided in the plotting
"PlottingTools" module. The core is implemented in BasePlotter.py which provides the basic interface.
In the context of plotting each object which can be plotted, e.g. histograms (TH1, TH2), graphs (TGraph), profiles, etc.
is typically referred to as *plotable object* (a dedicated generic implementation is currently worked on).

The actual plotting is done via the Plotter class in Plotter.py. There are several dedicated plotting algorithms provided:

* ComparisonPlotter.py: Compare distributions from different sources
* EventComparisionPlotter.py: Plot event-by-event comparisons
* RatioPlotter.py: Plots ratios and adds them to canvases
* CorrelationPlotter.py: Plots correlations

Basic wrappers to manipulate


AnalysisTools
--------------

The AnalysisTools modules contains a variety of tools used in typical HEP data analysis projects. This included background
estimation tools, selection tools, fitting procedures and ML tools. A brief overview is given below and dedicated examples
are given in **this has to be done**