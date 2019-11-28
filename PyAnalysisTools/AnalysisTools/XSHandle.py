from __future__ import division

import re
from builtins import object
from builtins import str
from collections import OrderedDict

from past.utils import old_div

from PyAnalysisTools.base import _logger, InvalidInputError
from PyAnalysisTools.base.YAMLHandle import YAMLLoader


def get_xsec_weight(lumi, process, xs_handle, event_numbers):
    if isinstance(lumi, OrderedDict) or isinstance(lumi, dict):
        if process.mc_campaign is None or re.search(r'mc16[acde]$', process.mc_campaign) is None:
            _logger.error('Could not find MC campaign information, but lumi was provided per MC '
                          'campaign. Not clear what to do. It will be assumed that you meant to scale '
                          'to total lumi. Please update and acknowledge once.')
            eval(input('Hit enter to continue or Ctrl+c to quit...'))
            lumi = sum(lumi.values())
        else:
            lumi = lumi[process.mc_campaign]
    cross_section_weight = xs_handle.get_lumi_scale_factor(process.process_name, lumi,
                                                           event_numbers[process])
    return cross_section_weight


class Dataset(object):
    """
    Representation of a single dataset (data or MC)
    """
    def __init__(self, **kwargs):
        kwargs.setdefault("is_data", False)
        kwargs.setdefault("is_mc", False)
        for k, v in list(kwargs.items()):
            setattr(self, k.lower(), v)
        if "cross_section" in kwargs and "*" in str(kwargs["cross_section"]):
            self.cross_section = eval(kwargs["cross_section"])


class XSInfo(object):
    """
    Cross-section summary object. Contains data/MC info, cross-section, k-factor, filter-efficiency
    """
    def __init__(self, dataset):
        if dataset.is_data:
            return
        for attr, val in list(dataset.__dict__.items()):
            setattr(self, attr, val)
        self.xsec = self.cross_section
        if not hasattr(dataset, 'kfactor'):
            self.kfactor = 1.
        if not hasattr(dataset, 'filtereff'):
            self.filtereff = 1.


class XSHandle(object):
    """
    Interface for cross-section scaling
    """
    def __init__(self, cross_section_file, read_dsid=False):
        if cross_section_file is None:
            self.invalid = True
            return
        self.invalid = False
        _logger.debug("XSHandle read cross section file {:s}".format(cross_section_file))
        cross_sections = YAMLLoader.read_yaml(cross_section_file)
        if not read_dsid:
            self.cross_sections = {value.process_name: XSInfo(value)
                                   for value in list(cross_sections.values()) if value.is_mc}
        else:
            self.cross_sections = {key: XSInfo(val)
                                   for key, val in list(cross_sections.items()) if val.is_mc}

    def get_xs_scale_factor(self, process):
        """
        Retrieve cross-section scale factor for a single process calculated based on PMG (or custom) cross section,
        filter-efficiency and k-factor (last two are optional)
        :param process: process name
        :type process: str
        :return: cross section scale factor
        :rtype: float
        """
        if self.invalid:
            return 1.
        if process not in self.cross_sections:
            _logger.error("Could not found process %s in cross section configuration" % process)
            return 1.
        xs_info = self.cross_sections[process]
        xs_scale_factor = float(xs_info.xsec)
        if hasattr(xs_info, "filtereff"):
            xs_scale_factor *= xs_info.filtereff
        # TODO: temporary fix
        # if hasattr(xs_info, "kfactor"):
        #     xs_scale_factor *= xs_info.kfactor
        return xs_scale_factor

    def retrieve_xs_info(self, process):
        """
        Get cross-section information (xsec, k-factor, filter-efficiency) for a single process
        :param process: process name
        :type process: str
        :return: cross-section, filter-efficiency, k-factor
        :rtype: tuple(float)
        """
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

    def get_lumi_scale_factor(self, process, lumi, mc_events, fixed_xsec=None):
        """
        Retrieve scale factor to scale process to specific luminosity
        :param process: process name
        :type process: str
        :param lumi: luminosity
        :type lumi: float
        :param mc_events: number of initially produced MC events (actually sum of weights)
        :type mc_events: float
        :param fixed_xsec: use fixed cross section instead of stored for process (optional) - useful for limit setting
        :type fixed_xsec: float
        :return: luminosity scale factor
        :rtype: float
        """
        if fixed_xsec:
            xsec = fixed_xsec
        else:
            xsec = self.get_xs_scale_factor(process)
        return old_div(xsec * lumi * 1000. * 1000., mc_events)

    def get_ds_info(self, process, element):
        """
        Retrieve a configurable attribute from a given process
        :param process: process name
        :type process: str
        :param element: attribute name (e.g. label, name)
        :type element: str
        :return: attribute
        :rtype: type(attribute)
        """
        try:
            return getattr(self.cross_sections[process], element)
        except KeyError:
            _logger.error('Cannot find process {:s} in cross section table'.format(process))
        except AttributeError:
            _logger.error('Cannot find element {:s} for process {:s}'.format(element, process))
        return None
