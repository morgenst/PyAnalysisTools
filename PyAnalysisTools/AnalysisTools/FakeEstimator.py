from PyAnalysisTools.base import _logger, InvalidInputError
from PyAnalysisTools.PlottingUtils.Plotter import Plotter
from PyAnalysisTools.ROOTUtils.FileHandle import FileHandle
from PyAnalysisTools.PlottingUtils.PlotConfig import parse_and_build_plot_config
import pathos.multiprocessing as mp


class MuonFakeEstimator(object):
    def __init__(self, **kwargs):
        if not "input_files" in kwargs:
            raise InvalidInputError("No input files provided")
        self.file_handles = [FileHandle(file_name=file_name) for file_name in kwargs["input_files"]]
        self.tree_name = kwargs["tree_name"]
        print kwargs["plot_config"]
        self.plot_config = parse_and_build_plot_config(kwargs["plot_config"])
        self.output_path = kwargs["output_path"]
        self.ncpu = 1
        self.plotter = Plotter(plot_config_files=[kwargs["plot_config"]], **kwargs)

    def plot_jet_bins(self):
        plot_config = filter(lambda pc: pc.name == "numerator_pt", self.plot_config[0])[0]
        for n_jet in range(3):
            plot_config.cut = "jet_n == {:d}".format(n_jet)
            muon_pt = mp.ThreadPool(min(self.ncpu, 1)).map(self.plotter.read_histograms,
                                                           [plot_config])
            print muon_pt[0], type(muon_pt[0])
            print muon_pt[0][1].GetEntries()


    def run(self):
        self.plot_jet_bins()
