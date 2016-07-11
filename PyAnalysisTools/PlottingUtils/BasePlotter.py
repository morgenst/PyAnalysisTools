__author__ = 'marcusmorgenstern'
__mail__ = ''

import ROOT
from PyAnalysisTools.base import _logger, InvalidInputError
from PyAnalysisTools.PlottingUtils.PlotConfig import parse_and_build_plot_config
from PyAnalysisTools.ROOTUtils.FileHandle import FileHandle
from PyAnalysisTools.PlottingUtils import Formatting as FM
from PyAnalysisTools.PlottingUtils import HistTools as HT


class BasePlotter(object):
    def __init__(self, **kwargs):
        if not "input_file" in kwargs:
            _logger.error("No input files provided")
            raise  InvalidInputError("No input files")
        if not "plot_config_file" in kwargs:
            _logger.error("No plot config file provided. Nothing to parse")
            raise InvalidInputError("No plot config file")
        for k,v in kwargs.iteritems():
            setattr(self, k, v)
        self.file_handle = FileHandle(self.input_file)

    def parse_plot_config(self):
        _logger.debug("Try to parse plot config file")
        self.plot_configs = parse_and_build_plot_config(self.plot_config_file)

    def initialise(self):
        self.parse_plot_config()

    @staticmethod
    def get_histogram_definition(plot_config):
        dimension = plot_config.dist.count(":")
        hist = None
        if dimension == 0:
            hist = ROOT.TH1F(plot_config.name, "", plot_config.bins, plot_config.xmin, plot_config.xmax)
        elif dimension == 1:
            hist = ROOT.TH2F(plot_config.name, "", plot_config.bins, plot_config.xmin, plot_config.xmax,
                             plot_config.ybins, plot_config.ymin, plot_config.ymax)
        elif dimension==2:
            hist = ROOT.TH3F(plot_config.name, "", plot_config.bins, plot_config.xmin, plot_config.xmax,
                             plot_config.ybins, plot_config.ymin, plot_config.ymax,
                             plot_config.zbins, plot_config.zmin, plot_config.zmax)
        if not hist:
            _logger.error("Unable to create histogram for plot_config %s for variable %s" % (plot_config.name,
                                                                                             plot_config.dist))
            raise InvalidInputError("Invalid plot configuration")
        return hist

    def make_plot(self, plot_config):
        hist = self.__class__.get_histogram_definition(plot_config)
        print hist
        try:
            self.file_handle.fetch_and_link_hist_to_tree(self.tree_name, hist, plot_config.dist, plot_config.cuts)
        except Exception as e:
            raise e
        self.format_hist(hist, plot_config)
        hist.Draw()
        raw_input()

    def format_hist(self, hist, plot_config):
        if hasattr(plot_config, "xtitle"):
            if hasattr(plot_config, "unit"):
                plot_config.xtitle += " [" + plot_config.unit + "]"
            FM.set_title_x(hist, plot_config.xtitle)
        y_title = "Entries"
        if hasattr(plot_config, "ytitle"):
            y_title = plot_config.ytitle
        if hasattr(plot_config, "unit"):
            y_title += " / %.1f %s" % (hist.GetXaxis().GetBinWidth(0), plot_config.unit)
        FM.set_title_y(hist, y_title)
        if hasattr(plot_config, "rebin"):
            HT.rebin(hist, plot_config.rebin)

    def make_plots(self):
        for plot_config in self.plot_configs:
            self.make_plot(plot_config)
