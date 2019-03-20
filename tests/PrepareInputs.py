import os
import ROOT

cwd = os.path.dirname(__file__)


class TestFileCreator(object):
    def __init__(self):
        self.f = ROOT.TFile.Open(os.path.join(cwd, "test.root"), "RECREATE")

    def save_to_file(self, obj):
        self.f.cd()
        obj.Write()

    def add_hist(self):
        test_hist = ROOT.TH1F("test_hist_1", "", 100, -1., 1.)
        test_hist.FillRandom("gaus", 10000)
        self.save_to_file(test_hist)


def prepare_cutflow_input():
    f = ROOT.TFile.Open("CutflowTestInput.root", "RECREATE")
    cutflow_raw = ROOT.TH1I("cutflow_raw", "cutflow_raw", 10, 0, 10)
    for i in range(10):
        cutflow_raw.SetBinContent(i, pow(10 - i, 2))
        cutflow_raw.GetXaxis().SetBinLabel(i + 1, "cut_%i" % i)
    f.cd()
    cutflow_raw.Write()
    f.Close()


if __name__ == '__main__':
    prepare_cutflow_input()
    creator = TestFileCreator()
    creator.add_hist()

