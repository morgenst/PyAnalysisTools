import ROOT


class EfficiencyCalculator(object):
    def __init__(self):
        pass

    def calculate_efficiency(self, numerator, denominator, name=None):
        """
        API for efficiency calculation. Depending on input type calculates 1D or 2D efficiency

        :rtype: efficiency histogram (TEfficiency for 1D, TH2F for 2D)
        """
        if isinstance(numerator, ROOT.TH1F):
            return self.calculate_1d_efficiency(numerator, denominator, name)
        elif isinstance(numerator, ROOT.TH2F):
            return self.calculate_2d_efficiency(numerator, denominator, name)

    @staticmethod
    def calculate_1d_efficiency(numerator, denominator, name=None):
        efficiency = ROOT.TEfficiency(numerator, denominator)
        if name:
            efficiency.SetName(name)
        return efficiency

    @staticmethod
    def calculate_2d_efficiency(numerator, denominator, name=None):
        """
        Calculate 2D efficiencies

        :param numerator: numerator hist
        :type numerator: TH2
        :param denominator: numerator hist
        :type denominator: TH2
        :param name: Name of efficiency object (optional)
        :type name: string
        :return: efficiency
        :rtype: TH2
        """
        efficiency = numerator.Clone("eff_{:s}".format((numerator.GetName())))
        efficiency.Divide(denominator)
        if name:
            efficiency.SetName(name)
        return efficiency
