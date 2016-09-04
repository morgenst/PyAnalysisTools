import ROOT


def set_batch_mode(enable=True):
    ROOT.gROOT.SetBatch(enable)
