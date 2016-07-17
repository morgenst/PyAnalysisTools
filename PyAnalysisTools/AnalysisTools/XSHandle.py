from PyAnalysisTools.base.YAMLHandle import YAMLLoader
from PyAnalysisTools.base import _logger

class XSHandle(object):
    def __init__(self, cross_section_file):
        if cross_section_file is None:
            self.invalid = True
            return
        self.invalid = False
        self.cross_sections = YAMLLoader().read_yaml(cross_section_file)

    def get_lumi_scale_factor(self, process):
        if self.invalid:
            return 1.
        if process not in self.cross_sections:
            _logger.error("Could not found process %s in cross section configuration" % process)
            return 1.
        xs_scale_factor = 1.
        return xs_scale_factor