from __future__ import absolute_import

import ROOT
from .Formatting import load_atlas_style

load_atlas_style()


def set_batch_mode(enable=True):
    ROOT.gROOT.SetBatch(enable)
