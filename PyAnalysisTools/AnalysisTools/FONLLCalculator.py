from PyAnalysisTools.ROOTUtils.FileHandle import FileHandle
from PyAnalysisTools.PlottingUtils.PlotConfig import PlotConfig as pc
from PyAnalysisTools.PlottingUtils.ComparisonPlotter import ComparisonPlotter


class FONLLCalculator(ComparisonPlotter):
    def __init__(self, **kwargs):
        self.file_handles = [FileHandle(file_name=fn) for fn in kwargs["input_files"]]
        self.fonll_prediction = self.parse_input_file(kwargs["fonll_file"])
        self.fonll_pc = pc(name="fonll_truth_muon_pt", dist="", xtitle="Truth #mu p_{T} [GeV]",
                           ytitle="d#sigma/dp_{T} [pb/GeV]", draw="HIST", logy=True, ymin=0.0001, ymax=1000000)
        kwargs.setdefault("plot_config_files", [])
        kwargs.setdefault("nfile_handles", 1)
        super(ComparisonPlotter, self).__init__(**kwargs)



    @staticmethod
    def parse_input_file(input_file):
        parsed_info = []
        with open(input_file) as f:
            for line in f.readlines():
                if line.startswith("#"):
                    continue
                line = line.replace("\n", "")
                parsed_info.append(map(float, line.split()))
        return parsed_info

    def make_plots(self):
        plot_config = pc(name="truth_muon_pt", dist="", xtitle="Truth #mu p_{T} [GeV]",
                         ytitle="d#sigma/dp_{T} [pb/GeV]", draw="HIST", logy=True, bins=100, xmin=0, xmax=300.,
                         ymin=0.0001, ymax=1000000)
        print self.file_handles
        pythia_hists = self.read_histograms(plot_config, self.file_handles)
        print pythia_hists