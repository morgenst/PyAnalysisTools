import ROOT
from copy import deepcopy
from PyAnalysisTools.base import _logger
from PyAnalysisTools.AnalysisTools.RegionBuilder import RegionBuilder
from PyAnalysisTools.PlottingUtils.PlotConfig import PlotConfig
from PyAnalysisTools.PlottingUtils.BasePlotter import BasePlotter


class RegionSummaryModule(object):
    def __init__(self, **kwargs):
        self.file_handles = kwargs["file_handles"]
        self.tree_name = kwargs["tree_name"]
        self.regions = RegionBuilder(**kwargs["region_config"]).regions
        self.hist = self.build_hist()
        self.plot_config = kwargs["plot_configs"][0]
        self.type = "HistFetching"
        self.pc = PlotConfig(name="weighted_yield", dist="weight > 0", weight="weight",  bins=1, xmin=0, xmax=100000)
        self.plotter = BasePlotter(input_files=[], plot_config_files=[], **kwargs)

    def build_hist(self):
        hist = ROOT.TH1F("region_summary", "", len(self.regions), 0., len(self.regions))
        ROOT.SetOwnership(hist, False)
        for region in self.regions:
            hist.GetXaxis().SetBinLabel(self.regions.index(region) + 1, region.label)
        return hist

    def get_hist(self, process_name):
        return self.hist.Clone("region_summary_{:s}".format(process_name))

    def fetch(self):
        histograms = []
        for file_handle in self.file_handles:
            yield_hist = self.get_hist(file_handle.process)
            for region in self.regions:
                self.pc.cuts = [region.convert2cut_string()]
                pc = deepcopy(self.pc)
                pc.name += region.name
                _, _, hist = self.plotter.fetch_histograms((file_handle, pc))
                if hist is None:
                    _logger.error("Could not retrieve hist from {:s}".format(file_handle.file_name))
                    continue
                yield_hist.Fill(region.label, hist.Integral())
            histograms.append((self.plot_config, file_handle.process, yield_hist))

        return histograms
