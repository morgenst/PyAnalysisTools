__author__ = 'marcusmorgenstern'
__mail__ = ''

import ROOT
from PyAnalysisTools.base import _logger, InvalidInputError
from PyAnalysisTools.PlottingUtils.PlotConfig import parse_and_build_plot_config, parse_and_build_process_config
from PyAnalysisTools.ROOTUtils.FileHandle import FileHandle
from PyAnalysisTools.PlottingUtils import Formatting as FM
from PyAnalysisTools.PlottingUtils import HistTools as HT
from PyAnalysisTools.PlottingUtils import PlottingTools as PT
from PyAnalysisTools.AnalysisTools.XSHandle import XSHandle


class BasePlotter(object):
    def __init__(self, **kwargs):
        if "input_files" not in kwargs:
            _logger.error("No input files provided")
            raise  InvalidInputError("No input files")
        if "plot_config_file" not in kwargs:
            _logger.error("No plot config file provided. Nothing to parse")
            raise InvalidInputError("No plot config file")
        if "process_config_file" not in kwargs:
            _logger.warning("No process config file provided. Unable to read process specific options.")
        kwargs.setdefault("process_config_file", None)
        kwargs.setdefault("xs_config_file", None)
        kwargs.setdefault("batch", False)
        for k,v in kwargs.iteritems():
            setattr(self, k, v)
        ROOT.gROOT.SetBatch(self.batch)
        self.file_handles = [FileHandle(input_file) for input_file in self.input_files]
        self.xs_handle = XSHandle()
        self.process_config = self.parse_process_config()
        FM.load_atlas_style()
        self.histograms = {}

    def parse_plot_config(self):
        _logger.debug("Try to parse plot config file")
        self.plot_configs, self.common_config = parse_and_build_plot_config(self.plot_config_file)

    def parse_process_config(self):
        if self.process_config_file is None:
            return None
        process_config = parse_and_build_process_config(self.process_config_file, self.xs_config_file)
        return process_config

    def initialise(self):
        self.parse_plot_config()

    @staticmethod
    def get_histogram_definition(plot_config):
        dimension = plot_config.dist.count(":")
        hist = None
        hist_name = plot_config.name
        if dimension == 0:
            hist = ROOT.TH1F(hist_name, "", plot_config.bins, plot_config.xmin, plot_config.xmax)
        elif dimension == 1:
            hist = ROOT.TH2F(hist_name, "", plot_config.bins, plot_config.xmin, plot_config.xmax,
                             plot_config.ybins, plot_config.ymin, plot_config.ymax)
        elif dimension==2:
            hist = ROOT.TH3F(hist_name, "", plot_config.bins, plot_config.xmin, plot_config.xmax,
                             plot_config.ybins, plot_config.ymin, plot_config.ymax,
                             plot_config.zbins, plot_config.zmin, plot_config.zmax)
        if not hist:
            _logger.error("Unable to create histogram for plot_config %s for variable %s" % (plot_config.name,
                                                                                             plot_config.dist))
            raise InvalidInputError("Invalid plot configuration")
        return hist

    def retrieve_histogram(self, file_handle, plot_config):
        hist = self.__class__.get_histogram_definition(plot_config)
        try:
            file_handle.fetch_and_link_hist_to_tree(self.tree_name, hist, plot_config.dist, plot_config.cuts)
            hist.SetName(hist.GetName() + "_" + file_handle.process)
        except Exception as e:
            raise e
        self.format_hist(hist, plot_config)
        return hist

    #todo: should be moved to formatter
    def format_hist(self, hist, plot_config):
        if hasattr(plot_config, "xtitle"):
            xtitle = plot_config.xtitle
            if hasattr(plot_config, "unit"):
                xtitle += " [" + plot_config.unit + "]"
            FM.set_title_x(hist, xtitle)
        y_title = "Entries"
        if hasattr(plot_config, "ytitle"):
            y_title = plot_config.ytitle
        if hasattr(plot_config, "unit"):
            y_title += " / %.1f %s" % (hist.GetXaxis().GetBinWidth(0), plot_config.unit)
        FM.set_title_y(hist, y_title)
        if hasattr(plot_config, "rebin"):
            HT.rebin(hist, plot_config.rebin)

    def make_plots(self):
        for file_handle in self.file_handles:
            for plot_config in self.plot_configs:
                try:
                    self.histograms[plot_config][file_handle.process] = self.retrieve_histogram(file_handle, plot_config)
                except KeyError:
                    self.histograms[plot_config] = {file_handle.process: self.retrieve_histogram(file_handle, plot_config)}

        for plot_config_name, data in self.histograms.iteritems():
            canvas = PT.plot_histograms(data, plot_config, self.common_config, self.process_config)
            canvas.Update()
            raw_input()
