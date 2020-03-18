from __future__ import print_function

from builtins import filter
from builtins import object

import PyAnalysisTools.PlottingUtils.HistTools as HT
import PyAnalysisTools.PlottingUtils.PlottingTools as pt
import ROOT
from PyAnalysisTools.AnalysisTools.RegionBuilder import RegionBuilder
from PyAnalysisTools.PlottingUtils.Plotter import Plotter
from PyAnalysisTools.base.FileHandle import FileHandle


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
        numerator_configs = [region for region in self.region_builder.regions if "numerator" in region.name]
        self.fake_configs = [(num_conf,
                              filter(lambda reg: reg.name == num_conf.name.replace("numerator", "denominator"),
                                     self.region_builder.regions)[0])
                             for num_conf in numerator_configs]

    def make_plots(self):
        fetched_histograms = self.plotter.read_histograms(file_handles=self.file_handles,
                                                          plot_configs=self.plot_configs)
        fetched_histograms = [hist_set for hist_set in fetched_histograms if all(hist_set)]
        self.plotter.categorise_histograms(fetched_histograms)
        self.plotter.apply_lumi_weights(self.plotter.histograms)
        if self.plotter.process_configs is not None:
            self.plotter.merge_histograms()
        histograms = self.plotter.histograms
        for plot_config, data in list(histograms.items()):
            plot_config.ignore_rebin = True
            self.plotter.make_plot(plot_config, data)

    def subtract_prompt(self):
        data_hists = {}
        for plot_config, data in list(self.plotter.histograms.items()):
            data_hists[plot_config] = data["Data"]
            for process, hist in list(data.items()):
                if process == "Data":
                    continue
                data_hists[plot_config].Add(hist, -1.0)
            if plot_config.rebin is not None:
                data_hists[plot_config] = HT.rebin(data_hists[plot_config], plot_config.rebin,
                                                   plot_config.disable_bin_width_division)
        return data_hists

    def calculate_fake_factors(self, numerator, denominator):
        pc, numerator_hist = numerator
        denominator = denominator[1]
        if hasattr(pc, "rebin") and pc.rebin is not None:
            numerator_hist = HT.rebin(numerator_hist, pc.rebin, pc.disable_bin_width_division)
            denominator = HT.rebin(denominator, pc.rebin, pc.disable_bin_width_division)
        numerator_hist.Divide(denominator)
        pc.name = pc.name.replace("numerator", "fake_factor")
        print(numerator_hist)
        if not isinstance(numerator_hist, ROOT.TH2F):
            pc.draw_option = "p"
            pc.ytitle = "Fake factor"
            pc.logy = False
            pc.ymin = 0.
            pc.ymax = 2.
        else:
            print("go to else")
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
            for numerator in [t for t in iter(list(fake_contributions.items())) if fake_def[0].name in t[0].name]:
                denominator = filter(lambda t: numerator[0].name.replace("numerator", "denominator") in t[0].name,
                                     iter(list(fake_contributions.items())))[0]
                self.calculate_fake_factors(numerator, denominator)
        self.plotter.output_handle.write_and_close()
