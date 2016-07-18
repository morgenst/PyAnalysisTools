from PyAnalysisTools.base.YAMLHandle import YAMLLoader
from PyAnalysisTools.base import _logger


class XSInfo(object):
    def __init__(self, **kwargs):
        for k,v in kwargs.iteritems():
            setattr(self, k.lower(), v)


class XSHandle(object):
    def __init__(self, cross_section_file):
        if cross_section_file is None:
            self.invalid = True
            return
        self.invalid = False
        self.cross_sections = {key: XSInfo(**value) for (key, value) in YAMLLoader().read_yaml(cross_section_file).iteritems()}

    def get_xs_scale_factor(self, process):
        if self.invalid:
            return 1.
        if process not in self.cross_sections:
            _logger.error("Could not found process %s in cross section configuration" % process)
            return 1.
        xs_info = self.cross_sections[process]
        xs_scale_factor = float(xs_info.xsec)
        if hasattr(xs_info, "filtereff"):
            xs_scale_factor *= xs_info.filtereff
        if hasattr(xs_info, "kfactor"):
            xs_scale_factor *= xs_info.kfactor
        return xs_scale_factor

    def get_lumi_scale_factor(self, process, lumi, mc_events):
        return self.get_xs_scale_factor(process) * lumi / mc_events
