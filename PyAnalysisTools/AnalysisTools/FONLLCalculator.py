import ROOT
from PyAnalysisTools.base.FileHandle import FileHandle
from PyAnalysisTools.PlottingUtils.PlotConfig import PlotConfig as pc
from PyAnalysisTools.PlottingUtils.ComparisonPlotter import ComparisonPlotter
from PyAnalysisTools.base.OutputHandle import OutputFileHandle


class FONLLCalculator(ComparisonPlotter):
    def __init__(self, **kwargs):
        self.file_handles = [FileHandle(file_name=fn) for fn in kwargs["input_files"]]
        self.fonll_prediction = self.parse_input_file(kwargs["fonll_file"])
        self.rebin = [14 + 2 * i for i in range(14)] + [45 + 5 * i for i in range(6)] + [80, 90, 100, 130, 160, 190,
                                                                                         250, 300]
        self.fonll_pc = pc(name="fonll_truth_muon_pt", dist="", xtitle="Truth #mu p_{T} [GeV]",
                           ytitle="d#sigma/dp_{T} [pb/GeV]", draw="E3", logy=True, ymin=0.0001, ymax=1000000, xmin=15.,
                           normalise=True, color=ROOT.kRed, normalise_range=[1, -1], style=1001, rebin=self.rebin)
        kwargs.setdefault("plot_config_files", [])
        kwargs.setdefault("nfile_handles", 1)
        self.fonll_process = "B(b)#rightarrow e (FONLL)" if "B2e" in kwargs["fonll_file"] \
            else "D(c)#rightarrow e (FONLL)"
        super(FONLLCalculator, self).__init__(reference_merge_file=kwargs["process_merge"],
                                              xs_config_file=kwargs["dataset_info_file"],
                                              **kwargs)
        self.color_palette = [ROOT.kRed, ROOT.kBlue, ROOT.kGreen, ROOT.kCyan, ROOT.kPink, ROOT.kOrange, ROOT.kBlue-4,
                              ROOT.kRed+3, ROOT.kGreen-2]
        self.output_handle = OutputFileHandle(overload="comparison", output_file_name="Compare.root", **kwargs)

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

    def get_fonll_hist(self):
        data = self.fonll_prediction
        hist = ROOT.TH1F("fonll", "", int(len(data) / 2.), 15., data[-1][0])
        for pt_bin in data:
            if pt_bin[0] < 15.:
                continue
            hist.Fill(pt_bin[0], pt_bin[1])
            uncertainty = (pt_bin[3] + pt_bin[2]) / 2.
            b = hist.FindBin(pt_bin[0])
            hist.SetBinError(b, uncertainty)
        return hist

    def make_plots(self):
        fonll_hist = self.get_fonll_hist()
        ratio_config = {"name": "ratio_truth_muon_pt", "ytitle": "FONLL/Pythia", "xmin": 15., "ymin": 0.7, "ymax": 2.,
                        "draw": "Marker"}
        plot_config = pc(name="truth_muon_pt", dist="truth_muon_pt / 1000.", xtitle="Truth #mu p_{T} [GeV]",
                         ytitle="Normalised", draw="HIST", logy=True, bins=int(len(self.fonll_prediction)/2.), xmin=15.,
                         xmax=self.fonll_prediction[-1][0], logx=True, ymin=1e-8, ymax=1.1, normalise=True,
                         ratio_config=ratio_config, normalise_range=[1, -1], rebin=self.rebin)
        pythia_hists = self.read_histograms(plot_config, self.file_handles)
        self.make_comparison_plot(plot_config, [dict(pythia_hists[1]), dict([(self.fonll_process, fonll_hist)])],
                                  self.fonll_pc)
        self.output_handle.write_and_close()
