import ROOT


class EfficiencyCalculator(object):
    def __init__(self):
        pass

    def calculate_efficiency(self, numerator, denominator, name=None):
        efficiency = ROOT.TEfficiency(numerator, denominator)
        if name:
            efficiency.SetName(name)
        return efficiency
