import ROOT
import traceback
from array import array
from itertools import permutations
from copy import copy, deepcopy
from functools import partial
from PyAnalysisTools.base import _logger, InvalidInputError, Utilities
import PyAnalysisTools.PlottingUtils.PlottingTools as pt
import PyAnalysisTools.PlottingUtils.HistTools as HT
import PyAnalysisTools.PlottingUtils.Formatting as FT
from PyAnalysisTools.AnalysisTools.RegionBuilder import RegionBuilder
from PyAnalysisTools.PlottingUtils.Plotter import Plotter
from PyAnalysisTools.ROOTUtils.FileHandle import FileHandle
from PyAnalysisTools.PlottingUtils.PlotConfig import find_process_config, PlotConfig
from PyAnalysisTools.ROOTUtils.ObjectHandle import get_objects_from_canvas_by_type, get_objects_from_canvas_by_name
from PyAnalysisTools.base.OutputHandle import OutputFileHandle
import pathos.multiprocessing as mp


class MuonFakeCalculator(object):
    def __init__(self, **kwargs):
        kwargs.setdefault("control_plots", False)
        kwargs.setdefault("ncpu", 5)
        kwargs.setdefault("nfile_handles", 5)
        self.plotter = Plotter(plot_config_files=[kwargs["plot_config"]], **kwargs)
        self.plot_config = self.plotter.plot_configs
        self.plotter.read_cutflows()
        self.tree_name = kwargs["tree_name"]
        self.histograms = {}
        self.file_handles = [FileHandle(file_name=file_name, dataset_info=kwargs["xs_config_file"])
                             for file_name in kwargs["input_files"]]
        for mod in self.plotter.modules_pc_modifiers:
            self.plot_configs = mod.execute(self.plotter.plot_configs)
        self.region_builder = filter(lambda mod: isinstance(mod, RegionBuilder), self.plotter.modules_pc_modifiers)[0]
        numerator_configs = filter(lambda region: "numerator" in region.name, self.region_builder.regions)
        self.fake_configs = [(num_conf,
                              filter(lambda reg: reg.name == num_conf.name.replace("numerator", "denominator"),
                                     self.region_builder.regions)[0])
                             for num_conf in numerator_configs]

    def make_plots(self):
        fetched_histograms = self.plotter.read_histograms(file_handle=self.file_handles, plot_configs=self.plot_configs)
        fetched_histograms = filter(lambda hist_set: all(hist_set), fetched_histograms)
        self.plotter.categorise_histograms(fetched_histograms)
        self.plotter.apply_lumi_weights(self.plotter.histograms)
        if self.plotter.process_configs is not None:
            self.plotter.merge_histograms()
        histograms = self.plotter.histograms
        for plot_config, data in histograms.iteritems():
            plot_config.ignore_rebin = True
            self.plotter.make_plot(plot_config, data)

    def subtract_prompt(self):
        data_hists = {}
        for plot_config, data in self.plotter.histograms.iteritems():
            data_hists[plot_config] = data["Data"]
            for process, hist in data.iteritems():
                if process == "Data":
                    continue
                data_hists[plot_config].Add(hist, -1.0)
            if plot_config.rebin is not None:
                data_hists[plot_config] = HT.rebin(data_hists[plot_config], plot_config.rebin)
        return data_hists

    def calculate_fake_factors(self, numerator, denominator):
        pc, numerator_hist = numerator
        denominator = denominator[1]
        if hasattr(pc, "rebin") and pc.rebin is not None:
            numerator_hist = HT.rebin(numerator_hist, pc.rebin)
            denominator = HT.rebin(denominator, pc.rebin)
        numerator_hist.Divide(denominator)
        pc.name = pc.name.replace("numerator", "fake_factor")
        print numerator_hist
        if not isinstance(numerator_hist, ROOT.TH2F):
            pc.draw_option = "p"
            pc.ytitle = "Fake factor"
            pc.logy = False
            pc.ymin = 0.
            pc.ymax = 2.
        else:
            print "go to else"
            pc.draw_option = "COLZ"
            pc.ztitle = "Fake factor"
            pc.logy = False
            pc.zmin = 0.
            pc.zmax = 2.
        canvas = pt.plot_obj(numerator_hist, pc)
        self.plotter.output_handle.register_object(canvas)

    def run(self):
        self.make_plots()
        fake_contributions = self.subtract_prompt()
        for fake_def in self.fake_configs:
            for numerator in filter(lambda t: fake_def[0].name in t[0].name, fake_contributions.iteritems()):
                denominator = filter(lambda t: numerator[0].name.replace("numerator", "denominator") in t[0].name,
                                     fake_contributions.iteritems())[0]
                self.calculate_fake_factors(numerator, denominator)
        self.plotter.output_handle.write_and_close()
