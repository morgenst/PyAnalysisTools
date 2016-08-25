import ROOT


class EfficiencyCalculator(object):
    def __init__(self):
        pass

    def calculate_efficiency(self, numerator, denomirator, name=None):
        efficiency = ROOT.TEfficiency(numerator, denomirator)
        if name:
            efficiency.SetName(name)
        return efficiency