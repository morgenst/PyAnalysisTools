from builtins import filter
from builtins import map
from builtins import range
from builtins import object
from copy import deepcopy
from itertools import combinations
from operator import itemgetter

from PyAnalysisTools.base import _logger, InvalidInputError
from PyAnalysisTools.base.FileHandle import FileHandle
from PyAnalysisTools.PlottingUtils.PlotConfig import PlotConfig, parse_and_build_plot_config, get_histogram_definition,\
    parse_and_build_process_config
import PyAnalysisTools.PlottingUtils.PlottingTools as pt
from PyAnalysisTools.base.OutputHandle import OutputFileHandle
from PyAnalysisTools.PlottingUtils.PlotConfig import find_process_config
import PyAnalysisTools.PlottingUtils.Formatting as fm


class CorrelationPlotter(object):
    def __init__(self, **kwargs):
        """
        Constructor

        :param kwargs: named arguments
        :type kwargs: dict
        """
        if "input_files" not in kwargs:
            _logger.error("No input file provided")
            InvalidInputError("Missing input file")
        if "tree_name" not in kwargs:
            _logger.error("No tree name provided")
            InvalidInputError("Missing tree name")
        kwargs.setdefault("xs_config_file", None)
        kwargs.setdefault("process_config_file", None)
        kwargs.setdefault("friend_directory", None)
        kwargs.setdefault("friend_tree_names", None)
        kwargs.setdefault("friend_file_pattern", None)
        kwargs.setdefault("store_all", not kwargs["disable_intermediate_plots"])
        self.file_handles = [FileHandle(file_name=fn, dataset_info=kwargs["xs_config_file"],
                                        friend_directory=kwargs["friend_directory"],
                                        friend_tree_names=kwargs["friend_tree_names"],
                                        friend_pattern=kwargs["friend_file_pattern"]) for fn in kwargs["input_files"]]
        self.tree_name = kwargs["tree_name"]
        for k, v in list(kwargs.items()):
            if k in ["input_files", "tree"]:
                continue
            setattr(self, k.lower(), v)
        self.process_configs = self.parse_process_config()

        self.variable_pcs, common_config = parse_and_build_plot_config(kwargs["variable_list"])
        self.output_handle = OutputFileHandle(output_dir=kwargs["output_path"], make_plotbook=True, set_title_name=True)
        self.correlation_hists = {}
        self.build_correlation_plot_configs()
        self.expand_process_configs()
        self.correlation_coeff_hist = None
        self.correlation_coeff_hists = {}

    def parse_process_config(self):
        if self.process_config_file is None:
            return None
        process_config = parse_and_build_process_config(self.process_config_file)
        return process_config

    def expand_process_configs(self):
        if self.process_configs is not None:
            for fh in self.file_handles:
                _ = find_process_config(fh.process, self.process_configs)

    def __del__(self):
        self.output_handle.write_and_close()

    def fill_correlation_coefficient(self, coefficient, var_x, var_y, process):
        self.correlation_coeff_hists[process].Fill(var_x, var_y, coefficient)
        if not var_x == var_y:
            self.correlation_coeff_hists[process].Fill(var_y, var_x, coefficient)

    def fetch_correlation_hist(self, fh, plot_config):
        pc = deepcopy(plot_config)
        pc.name = "{:s}_{:s}".format(pc.name, fh.process_with_mc_campaign)
        fh.reset_friends()
        hist = get_histogram_definition(pc)
        fh.fetch_and_link_hist_to_tree(self.tree_name, hist, pc.dist, tdirectory="Nominal")
        hist.SetDirectory(0)
        try:
            self.correlation_hists[fh.process_with_mc_campaign].append(hist)
        except KeyError:
            self.correlation_hists[fh.process_with_mc_campaign] = [hist]
        if self.store_all:
            self.make_correlation_plot(hist, pc)
            correlation_coefficient = hist.GetCorrelationFactor()
            var_x, var_y = plot_config.dist.split(":")
            self.fill_correlation_coefficient(correlation_coefficient, var_x, var_y, fh.process_with_mc_campaign)
        fh.release_friends()

    def make_correlation_plot(self, hist, plot_config):
        canvas = pt.plot_obj(hist, plot_config)
        self.output_handle.register_object(canvas)

    def merge_hists(self):
        if self.process_configs is None:
            return
        self.merge(self.correlation_hists, self.process_configs)

    def merge(self, histograms, process_configs):
        for process, process_config in list(process_configs.items()):

            if not hasattr(process_config, "subprocesses"):
                continue
            for sub_process in process_config.subprocesses:
                # TODO: this is a ridiculously stupid implementation
                if sub_process not in list(histograms.keys()) and sub_process + ".mc16a" not in list(
                        histograms.keys()) and sub_process + ".mc16d" not in list(histograms.keys()):
                    continue
                for extension in ["", ".mc16a", ".mc16c", ".mc16d"]:
                    tmp_sub_process = sub_process + extension
                    if process not in list(histograms.keys()):
                        if tmp_sub_process not in list(histograms.keys()):
                            continue
                        for hist in histograms[tmp_sub_process]:
                            new_hist_name = hist.GetName().replace(tmp_sub_process, process)
                            try:
                                histograms[process].append(hist.Clone(new_hist_name))
                            except Exception:
                                histograms[process] = [hist.Clone(new_hist_name)]
                    else:
                        if tmp_sub_process not in list(histograms.keys()):
                            continue
                        for index, hist in enumerate(histograms[tmp_sub_process]):
                            histograms[process][index].Add(hist)
                    histograms.pop(tmp_sub_process)

    def make_correlation_plots(self):
        def get_plot_config(hist):
            return filter(lambda pc: pc.name in hist.GetName(), self.corr_plot_configs)[0]

        for fh in self.file_handles:
            if fh.process_with_mc_campaign not in self.correlation_coeff_hists:
                self.prepare_correlation_coefficient_hist(self.variable_list, fh.process_with_mc_campaign)
            for combination in self.corr_plot_configs:
                self.fetch_correlation_hist(fh, combination)
        self.merge_hists()
        for process, hist_set in list(self.correlation_hists.items()):
            if process not in self.correlation_coeff_hists:
                self.prepare_correlation_coefficient_hist(self.variable_list, process)

            for hist in hist_set:
                pc = deepcopy(get_plot_config(hist))
                pc.name = hist.GetName()
                self.make_correlation_plot(hist, pc)
                correlation_coefficient = hist.GetCorrelationFactor()
                var_x, var_y = pc.dist.split(":")
                self.fill_correlation_coefficient(correlation_coefficient, var_x, var_y, process)

        for corr_coeff in list(self.correlation_coeff_hists.values()):
            pc = deepcopy(self.plot_config_corr_coeff)
            pc.name = corr_coeff.GetName()
            canvas = pt.plot_2d_hist(corr_coeff, pc)
            self.output_handle.register_object(canvas)

    def prepare_correlation_coefficient_hist(self, variables, process):
        bins = len(variables)
        name = "linear_correlation_coefficients_{:s}".format(process)
        if self.correlation_coeff_hist is not None:
            self.correlation_coeff_hists[process] = self.correlation_coeff_hist.Clone(name)
            return
        self.plot_config_corr_coeff = PlotConfig(name=name, dist=":", xtitle="", ytitle="",
                                                 ztitle="Lin. correlation coefficent",
                                                 xmin=0, xmax=bins, xbins=bins, ymin=0, ymax=bins, ybins=bins,
                                                 draw_option="COLZTEXT", zmin=-1., zmax=1.)
        self.correlation_coeff_hist = get_histogram_definition(self.plot_config_corr_coeff)
        self.correlation_coeff_hists[process] = self.correlation_coeff_hist
        for ibin in range(len(variables)):
            self.correlation_coeff_hists[process].GetXaxis().SetBinLabel(ibin + 1, variables[ibin])
            self.correlation_coeff_hists[process].GetYaxis().SetBinLabel(bins - ibin, variables[ibin])

    def build_correlation_plot_configs(self):
        def build_correlation_plot_config(comb):
            pc_var_x = comb[0]
            pc_var_y = comb[1]
            plot_config = PlotConfig(dist="%s:%s" % (pc_var_y.dist, pc_var_x.dist),
                                     name="correlation_%s_%s" % (pc_var_x.name, pc_var_y.name),
                                     xmin=pc_var_x.xmin, xmax=pc_var_x.xmax, xbins=pc_var_x.bins,
                                     ymin=pc_var_y.xmin, ymax=pc_var_y.xmax, ybins=pc_var_y.bins,
                                     draw_option="COLZ", xtitle=pc_var_x.xtitle, ytitle=pc_var_y.xtitle,
                                     watermark="Internal", ztitle="Entries")
            return plot_config

        self.variable_combinations = list(combinations(self.variable_pcs, 2))
        self.variable_list = [pc.dist for pc in self.variable_pcs]
        self.corr_plot_configs = [build_correlation_plot_config(comb) for comb in self.variable_combinations]

    def make_profile_plots(self):
        def get_plot_config(hist):
            return filter(lambda pc: "_".join(pc.name.split("_")[1:]) in hist.GetName(), self.corr_plot_configs)[0]

        def get_color(process):
            mc_campaign_pattern = ""
            if "mc16a" in process:
                mc_campaign_pattern = ".mc16a"
            elif "mc16c" in process:
                mc_campaign_pattern = ".mc16c"
            elif "mc16d" in process:
                mc_campaign_pattern = ".mc16d"
            if process not in self.process_configs:
                base_process = process.replace(mc_campaign_pattern, "")
                self.process_configs[process] = deepcopy(self.process_configs[base_process])
                self.process_configs[process].color = self.process_configs[process].color + " + {:d}".format(3)
                self.process_configs[process].label = "{:s} ({:s})".format(self.process_configs[process].label,
                                                                           mc_campaign_pattern.replace(".", ""))
            return self.process_configs[process].color

        profiles = {}
        for process, hists in list(self.correlation_hists.items()):
            for hist in hists:
                hist_base_name = "_".join(hist.GetName().split("_")[1:-1])
                profileX = hist.ProfileX(hist_base_name + "_profileX_{:s}".format(process))
                profileY = hist.ProfileY(hist_base_name + "_profileY_{:s}".format(process))
                if process not in profiles:
                    profiles[process] = [profileX]
                    profiles[process].append(profileY)
                    continue
                profiles[process].append(profileX)
                profiles[process].append(profileY)
        profile_plots = {}
        labels = []
        colors = [(process, get_color(process)) for process in list(profiles.keys())]
        marker_styles = list(range(20, 34))
        for process, hists in list(profiles.items()):
            labels.append(process)
            for hist in hists:
                pc = deepcopy(get_plot_config(hist))
                if "profileX" in hist.GetName():
                    pc.name = pc.name.replace("correlation", "profileX")
                    pc.ytitle = "<{:s}>".format(pc.ytitle)
                if "profileY" in hist.GetName():
                    pc.name = pc.name.replace("correlation", "profileY")
                    pc.xmin, pc.ymin = pc.ymin, pc.xmin
                    pc.xmax, pc.ymax = pc.ymax, pc.xmax
                    xtitle = "<{:s}>".format(pc.xtitle)
                    pc.xtitle = pc.ytitle
                    pc.ytitle = xtitle

                pc.draw = "Marker"
                pc.color = list(map(itemgetter(1), colors))
                index = map(itemgetter(0), colors).index(process)
                pc.style = marker_styles[index]
                hist_base_name = pc.name
                if hist_base_name not in profile_plots:
                    canvas = pt.plot_hist(hist, pc, index=index)
                    pc.lumi = self.lumi
                    fm.decorate_canvas(canvas, plot_config=pc)
                    profile_plots[hist_base_name] = canvas
                else:
                    pt.add_object_to_canvas(profile_plots[hist_base_name], hist, pc, index=index)
        list([fm.add_legend_to_canvas(c, ratio=False, process_configs=self.process_configs, format="marker")
              for c in list(profile_plots.values())])
        list([self.output_handle.register_object(c) for c in list(profile_plots.values())])
