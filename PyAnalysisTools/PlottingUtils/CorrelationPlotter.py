import ROOT
from itertools import combinations
from PyAnalysisTools.base import _logger, InvalidInputError
from PyAnalysisTools.ROOTUtils.FileHandle import FileHandle
from PyAnalysisTools.PlottingUtils.PlotConfig import PlotConfig
import PyAnalysisTools.PlottingUtils.PlottingTools as PT
from PyAnalysisTools.base.OutputHandle import OutputFileHandle


class CorrelationPlotter(object):
    def __init__(self, **kwargs):
        if not "input_file" in kwargs:
            _logger.error("No input file provided")
            InvalidInputError("Missing input file")
        if not "tree_name" in kwargs:
            _logger.error("No tree name provided")
            InvalidInputError("Missing tree name")
        self.file_handle = FileHandle(file_name=kwargs["input_file"])
        self.tree_name = kwargs["tree_name"]
        self.tree = self.file_handle.get_object_by_name(self.tree_name, tdirectory="Nominal")
        kwargs.setdefault("output_file", "correlation_%s.root" % self.file_handle.parse_process())
        for k, v in kwargs.iteritems():
            if k in ["input_files", "tree"]:
                continue
            setattr(self, k.lower(), v)
        self.output_handle = OutputFileHandle(self.output_file)

    # todo: should be move to more abstract class
    @staticmethod
    def get_histogram_definition(plot_config):
        dimension = plot_config.dist.count(":")
        hist = None
        hist_name = plot_config.name
        if dimension == 0:
            hist = ROOT.TH1F(hist_name, "", plot_config.bins, plot_config.xmin, plot_config.xmax)
        elif dimension == 1:
            hist = ROOT.TH2F(hist_name, "", plot_config.xbins, plot_config.xmin, plot_config.xmax,
                             plot_config.ybins, plot_config.ymin, plot_config.ymax)
        elif dimension == 2:
            hist = ROOT.TH3F(hist_name, "", plot_config.xbins, plot_config.xmin, plot_config.xmax,
                             plot_config.ybins, plot_config.ymin, plot_config.ymax,
                             plot_config.zbins, plot_config.zmin, plot_config.zmax)
        if not hist:
            _logger.error("Unable to create histogram for plot_config %s for variable %s" % (plot_config.name,
                                                                                             plot_config.dist))
            raise InvalidInputError("Invalid plot configuration")
        return hist

    def fill_correlation_coefficient(self, coefficient, var_x, var_y):
        self.correlation_coeff_hist.Fill(var_x, var_y, coefficient)
        if not var_x == var_y:
            self.correlation_coeff_hist.Fill(var_y, var_x, coefficient)

    def make_correlation_plot(self, combination):
        var_x, var_y = combination
        plot_config = PlotConfig(dist="%s:%s" % (var_x,var_y),
                                 name="correlation_%s_%s" % (var_x, var_y),
                                 xmin=self.tree.GetMinimum(var_x),
                                 xmax=self.tree.GetMaximum(var_x),
                                 xbins=100,
                                 ymin=self.tree.GetMinimum(var_y),
                                 ymax=self.tree.GetMaximum(var_y),
                                 ybins=100,
                                 draw="COLZ")
        hist = self.__class__.get_histogram_definition(plot_config)
        self.file_handle.fetch_and_link_hist_to_tree(self.tree_name, hist, plot_config.dist, tdirectory="Nominal")
        correlation_coefficient = hist.GetCorrelationFactor()
        self.fill_correlation_coefficient(correlation_coefficient, *combination)
        canvas = PT.plot_hist(hist, plot_config)
        self.output_handle.register_object(canvas)

    def make_correlation_plots(self, variables=None):
        if variables is None:
            variables = self.file_handle.get_branch_names_from_tree(self.tree_name)
        variable_combinations = list(combinations(variables, 2))
        self.prepare_correlation_coefficient_hist(variables)
        for combination in variable_combinations:
            self.make_correlation_plot(combination)
        for variable in variables:
            self.fill_correlation_coefficient(1., variable, variable)
        canvas = PT.plot_hist(self.correlation_coeff_hist, self.plot_config)
        self.output_handle.register_object(canvas)
        self.output_handle.write_and_close()

    def prepare_correlation_coefficient_hist(self, variables):
        bins=len(variables)
        self.plot_config = PlotConfig(name="linear_correlation_coefficients", dist=":",
                                      xmin=0, xmax=bins, xbins=bins, ymin=0, ymax=bins, ybins=bins, draw="COLZTEXT")
        self.correlation_coeff_hist = self.get_histogram_definition(self.plot_config)
        for ibin in range(len(variables)):
            self.correlation_coeff_hist.GetXaxis().SetBinLabel(ibin + 1, variables[ibin])
            self.correlation_coeff_hist.GetYaxis().SetBinLabel(bins - ibin, variables[ibin])
