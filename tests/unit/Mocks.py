import ROOT
from mock import MagicMock, Mock


def clone_and_rename(name='Clone'):
    h = ROOT.TH1F(name, '', 2, 0., 1.)
    return h

hist = Mock()
hist.GetName = MagicMock(return_value='foo')
hist.GetNbinsX = MagicMock(return_value=2)
hist.Clone = MagicMock(side_effect=clone_and_rename)
