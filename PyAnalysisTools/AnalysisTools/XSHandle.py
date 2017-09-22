from PyAnalysisTools.base.YAMLHandle import YAMLLoader
from PyAnalysisTools.base import _logger, InvalidInputError


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
        for attr, val in dataset.__dict__.iteritems():
            setattr(self, attr, val)
        self.xsec = self.cross_section
        if not hasattr(dataset, "kfactor"):
            self.kfactor = 1.
        if not hasattr(dataset, "filtereff"):
            self.filtereff = 1.


class XSHandle(object):
    def __init__(self, cross_section_file, read_dsid=False):
        if cross_section_file is None:
            self.invalid = True
            return
        self.invalid = False
        if not read_dsid:
            self.cross_sections = {value.process_name: XSInfo(value)
                                   for value in YAMLLoader.read_yaml(cross_section_file).values() if value.is_mc}
        else:
            self.cross_sections = {key: XSInfo(val)
                                   for key, val in YAMLLoader.read_yaml(cross_section_file).iteritems() if val.is_mc}

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

    def retrieve_xs_info(self, process):
        if self.invalid:
            raise InvalidInputError("Invalid config of XSHandle")
        xsec, filter_eff, kfactor = None, 1., 1.
        if process not in self.cross_sections:
            _logger.error("Could not found process %s in cross section configuration" % process)
            return xsec, filter_eff, kfactor
        xs_info = self.cross_sections[process]
        xsec = float(xs_info.xsec)
        if hasattr(xs_info, "filtereff"):
            filter_eff = xs_info.filtereff
        if hasattr(xs_info, "kfactor"):
            kfactor = xs_info.kfactor
        return xsec, filter_eff, kfactor

    def get_lumi_scale_factor(self, process, lumi, mc_events):
        return self.get_xs_scale_factor(process) * lumi * 1000. * 1000. / mc_events

    def get_ds_info(self, process, element):
        return getattr(self.cross_sections[process], element)
