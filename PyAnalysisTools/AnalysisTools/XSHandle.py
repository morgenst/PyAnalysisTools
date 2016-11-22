from PyAnalysisTools.base.YAMLHandle import YAMLLoader
from PyAnalysisTools.base import _logger


class Dataset(object):
    def __init__(self, **kwargs):
        kwargs.setdefault("is_data", False)
        kwargs.setdefault("is_mc", False)
        for k, v in kwargs.iteritems():
            setattr(self, k.lower(), v)
        if "cross_section" in kwargs and "*" in str(kwargs["cross_section"]):
            self.cross_section = eval(kwargs["cross_section"])


class XSInfo(object):
    def __init__(self, dataset):
        if dataset.is_data:
            return
        self.xsec = dataset.cross_section
        self.kfactor = 1.
        self.filtereff = 1.
        if hasattr(dataset, "kfactor"):
            self.kfactor = dataset.kfactor
        if hasattr(dataset, "filtereff"):
            self.filtereff = dataset.filtereff


class XSHandle(object):
    def __init__(self, cross_section_file):
        if cross_section_file is None:
            self.invalid = True
            return
        self.invalid = False
        self.cross_sections = {value.process_name: XSInfo(value)
                               for value in YAMLLoader.read_yaml(cross_section_file).values() if value.is_mc}

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
        return self.get_xs_scale_factor(process) * lumi * 1000. * 1000. / mc_events
