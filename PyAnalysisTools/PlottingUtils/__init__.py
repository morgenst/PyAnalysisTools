from __future__ import absolute_import

import ROOT


def set_batch_mode(enable=True):
    ROOT.gROOT.SetBatch(enable)
