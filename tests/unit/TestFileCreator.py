from builtins import object
import ROOT
import sys
import argparse


class TestFileCreator(object):
    def __init__(self):
        self.f = ROOT.TFile.Open("test.root", "RECREATE")

    def save_to_file(self, obj):
        self.f.cd()
        obj.Write()

    def add_hist(self):
        test_hist = ROOT.TH1F("test_hist_1", "", 100, -1., 1.)
        test_hist.FillRandom("gaus", 10000)
        self.save_to_file(test_hist)


def main(argv):
    parser = argparse.ArgumentParser(description="Helper to setup root test file for unittests")
    parser.parse_args()
    creator = TestFileCreator()
    creator.add_hist()


if __name__ == '__main__':
    main(sys.argv[1:])
