import ROOT
from copy import deepcopy
from itertools import combinations
from PyAnalysisTools.base import _logger, InvalidInputError
from PyAnalysisTools.ROOTUtils.FileHandle import FileHandle
from PyAnalysisTools.PlottingUtils.PlotConfig import PlotConfig, parse_and_build_plot_config, get_histogram_definition
import PyAnalysisTools.PlottingUtils.PlottingTools as pt
from PyAnalysisTools.base.OutputHandle import OutputFileHandle


class CorrelationPlotter(object):
    def __init__(self, **kwargs):
        """
        Constructor

        :param kwargs: named arguments
        :type kwargs: dict
        """
        if not "input_files" in kwargs:
            _logger.error("No input file provided")
            InvalidInputError("Missing input file")
        if not "tree_name" in kwargs:
            _logger.error("No tree name provided")
            InvalidInputError("Missing tree name")
        kwargs.setdefault("xs_config_file", None)
        kwargs.setdefault("store_all", True)
        self.file_handles = [FileHandle(file_name=fn,
                                        dataset_info=kwargs["xs_config_file"]) for fn in kwargs["input_files"]]
        self.tree_name = kwargs["tree_name"]
        #kwargs.setdefault("output_file", "correlation_%s.root" % self.file_handle.parse_process())
        for k, v in kwargs.iteritems():
            if k in ["input_files", "tree"]:
                continue
            setattr(self, k.lower(), v)
        self.variable_pcs, common_config = parse_and_build_plot_config(kwargs["variable_list"])
        self.output_handle = OutputFileHandle(output_dir=kwargs["output_path"])
        self.correlation_hists = {}
        self.build_correlation_plot_configs()

    def __del__(self):
        self.output_handle.write_and_close()

    def fill_correlation_coefficient(self, coefficient, var_x, var_y):
        self.correlation_coeff_hist.Fill(var_x, var_y, coefficient)
        if not var_x == var_y:
            self.correlation_coeff_hist.Fill(var_y, var_x, coefficient)

    def make_correlation_plot(self, fh, plot_config):
        pc = deepcopy(plot_config)
        hist = get_histogram_definition(pc)
        pc.name = "{:s}_{:s}".format(pc.name,fh.process)
        fh.fetch_and_link_hist_to_tree(self.tree_name, hist, pc.dist, tdirectory="Nominal")
        correlation_coefficient = hist.GetCorrelationFactor()
        #self.fill_correlation_coefficient(correlation_coefficient, *combination)
        try:
            self.correlation_hists[fh.process].append(hist)
        except KeyError:
            self.correlation_hists[fh.process] = [hist]
        canvas = pt.plot_hist(hist, pc)
        if self.store_all:
            self.output_handle.register_object(canvas)

    def make_correlation_plots(self, variables=None):
        for fh in self.file_handles:
            tree = fh.get_object_by_name(self.tree_name, tdirectory="Nominal")
            # if variables is None:
            #     variables = self.file_handle.get_branch_names_from_tree(self.tree_name)
            # variable_combinations = list(combinations(variables, 2))
            #self.prepare_correlation_coefficient_hist(variables)
            for combination in self.corr_plot_configs:
                self.make_correlation_plot(fh, combination)
        # for variable in variables:
        #     self.fill_correlation_coefficient(1., variable, variable)
        # canvas = PT.plot_hist(self.correlation_coeff_hist, self.plot_config)
        # self.output_handle.register_object(canvas)

    def prepare_correlation_coefficient_hist(self, variables):
        bins=len(variables)
        self.plot_config = PlotConfig(name="linear_correlation_coefficients", dist=":",
                                      xmin=0, xmax=bins, xbins=bins, ymin=0, ymax=bins, ybins=bins, draw="COLZTEXT")
        self.correlation_coeff_hist = self.get_histogram_definition(self.plot_config)
        for ibin in range(len(variables)):
            self.correlation_coeff_hist.GetXaxis().SetBinLabel(ibin + 1, variables[ibin])
            self.correlation_coeff_hist.GetYaxis().SetBinLabel(bins - ibin, variables[ibin])

    def build_correlation_plot_configs(self):
        def build_correlation_plot_config(comb):
            pc_var_x = comb[0]
            pc_var_y = comb[1]
            plot_config = PlotConfig(dist="%s:%s" % (pc_var_x.dist, pc_var_y.dist),
                                     name="correlation_%s_%s" % (pc_var_x.name, pc_var_y.name),
                                     xmin=pc_var_x.xmin, xmax=pc_var_x.xmax, xbins=pc_var_x.bins,
                                     ymin=pc_var_y.xmin, ymax=pc_var_y.xmax, ybins=pc_var_y.bins,
                                     draw="COLZ", xtitle=pc_var_x.xtitle, ytitle=pc_var_y.xtitle,
                                     watermark="Internal")
            return plot_config
        self.variable_combinations = list(combinations(self.variable_pcs, 2))
        self.corr_plot_configs = [build_correlation_plot_config(comb) for comb in self.variable_combinations]

    def make_profile_plots(self):
        def get_plot_config(hist):
            return filter(lambda pc: pc.name in hist.GetName(), self.variable_pcs)[0]
        profiles = {}
        for process, hists in self.correlation_hists.iteritems():
            for hist in hists:
                hist_base_name = "_".join(hist.GetName().split("_")[:-1])
                profile = hist.ProfileX(hist_base_name+"_profileX")
                if process not in profiles:
                    profiles[process] = [profile]
                    continue
                profiles[process].append(profile)
        profile_plots = {}
        for process, hists in profiles.iteritems():
            for hist in hists:
                pc = get_plot_config(hist)
                hist_base_name = "_".join(hist.GetName().split("_")[:-1])
                if hist_base_name not in profile_plots:
                    profile_plots[hist_base_name] = pt.plot_graph(hist, pc)
                else:
                    pt.add_object_to_canvas(profile_plots[hist_base_name], hist, pc)
        map(lambda c: self.output_handle.register_object(c), profile_plots.values())