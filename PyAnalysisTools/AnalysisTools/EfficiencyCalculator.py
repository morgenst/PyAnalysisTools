import ROOT


class EfficiencyCalculator(object):
    def __init__(self):
        pass

    def calculate_efficiency(self, numerator, denominator, name=None):
        if isinstance(numerator, ROOT.TH1F):
            return self.calculate_1d_efficiency(numerator, denominator, name)
        elif isinstance(numerator, ROOT.TH2F):
            return self.calculate_2d_efficiency(numerator, denominator, name)

    def calculate_1d_efficiency(self, numerator, denominator, name=None):
        efficiency = ROOT.TEfficiency(numerator, denominator)
        if name:
            efficiency.SetName(name)
        return efficiency

    def calculate_2d_efficiency(self, numerator, denominator, name=None):
        efficiency = numerator.Clone("eff_{:s}".format((numerator.GetName())))
        efficiency.Divide(denominator)
        if name:
            efficiency.SetName(name)
        return efficiency
