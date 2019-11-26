from __future__ import division
from __future__ import print_function

import os
import re
from array import array
from collections import OrderedDict
from copy import copy, deepcopy
import math
from math import log10

from builtins import input
from builtins import object
from builtins import range
from builtins import str

from past.utils import old_div

import ROOT
from PyAnalysisTools.PlottingUtils.HistTools import check_name
from PyAnalysisTools.base import _logger, InvalidInputError
from PyAnalysisTools.base.ProcessConfig import ProcessConfig, Process
from PyAnalysisTools.base.ShellUtils import find_file
from PyAnalysisTools.base.YAMLHandle import YAMLLoader as yl


class PlotConfig(object):
    def __init__(self, **kwargs):
        kwargs.setdefault('process_weight', None)
        kwargs.setdefault('name', 'default_plot_config')
        if "dist" not in kwargs and "is_common" not in kwargs:
            _logger.debug("Plot config does not contain distribution. Add dist key")
        if 'Lumi' in kwargs:
            _logger.error('Support for Lumi is dropped. Please use lumi keyword instead (small).')
            kwargs['lumi'] = kwargs['Lumi']
        kwargs.setdefault("cuts", None)
        # kwargs.setdefault("cuts_l1", None)
        if "draw" not in kwargs:
            kwargs.setdefault("Draw", "hist")
        user_config = find_file('plot_config_defaults.yml', os.path.join(os.curdir, '../'))
        py_ana_config_file_name = os.path.join(os.path.dirname(__file__), 'plot_config_defaults.yml')
        defaults_py_ana = yl.read_yaml(py_ana_config_file_name)
        usr_defaults = defaults_py_ana
        if user_config is not None:
            config_file_name = user_config
            usr_defaults = yl.read_yaml(config_file_name)

        for key, attr in list(defaults_py_ana.items()):
            if key in usr_defaults:
                attr = usr_defaults[key]
            if isinstance(attr, str):
                try:
                    kwargs.setdefault(key, eval(attr))
                except (NameError, SyntaxError):
                    kwargs.setdefault(key, attr)
            else:
                kwargs.setdefault(key, attr)

        for k, v in list(kwargs.items()):
            if v == "None":
                v = None
            if k == "ratio_config" and v is not None:
                v["logx"] = kwargs["logx"]
                self.set_additional_config("ratio_config", **v)
                continue
            if k == "significance_config":
                self.set_additional_config("significance_config", **v)
                continue
            if "xmin" in k or "xmax" in k:
                v = eval(str(v))
            if (k == "ymax" or k == "ymin") and v is not None:
                if isinstance(v, float):
                    setattr(self, k.lower(), v)
                    continue
                setattr(self, k.lower(), eval(v))
                continue
            setattr(self, k.lower(), v)
        if self.calcsig and self.ratio:
            _logger.error('Requested both significance calculation and ratio plotting which is currently not supported.'
                          'This requires an implementation of at least two ratio pads. Thus disabling calcsig')
            self.calcsig = False
        self.is_multidimensional = False
        self.auto_decorate()
        self.used_mc_campaigns = []

    def is_set_to_value(self, attr, value):
        """
        Checks if attribute is set to value
        :param attr: attribute
        :type attr: str
        :param value: value of attribute
        :type value: any
        :return: True/False
        :rtype: boolean
        """
        if not hasattr(self, attr):
            return False
        return getattr(self, attr) == value

    def set_additional_config(self, attr_name, **kwargs):
        kwargs.setdefault("name", "ratio")
        kwargs.setdefault("dist", "ratio")
        kwargs.setdefault("ignore_style", False)
        kwargs.setdefault("enable_legend", False)
        kwargs.setdefault("ignore_process_labels", False)
        setattr(self, attr_name, PlotConfig(**kwargs))

    def __str__(self):
        """
        Overloaded str operator. Get's called if object is printed
        :return: formatted string with name and attributes
        :rtype: str
        """
        obj_str = "Plot config: {:s} \n".format(self.name)
        for attribute, value in list(self.__dict__.items()):
            obj_str += '{}={} '.format(attribute, value)
        return obj_str

    def __eq__(self, other):
        """
        Comparison operator
        :param other: plot config object to compare to
        :type other: PlotConfig
        :return: True/False
        :rtype: boolean
        """
        if isinstance(self, other.__class__):
            return self.__dict__ == other.__dict__
        return False

    def __ne__(self, other):
        """
        Comparison operator (negative)
        :param other: plot config object to compare to
        :type other: PlotConfig
        :return: True/False
        :rtype: boolean
        """
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.name)

    @staticmethod
    def get_overwritable_options():
        """
        Get properties which can be overwriten by specific plot config

        :return: overwritable properties
        :rtype: list
        """
        return ["outline", "make_plot_book", "no_data", "draw", "ordering", "signal_scale", "lumi", "normalise",
                "merge_mc_campaigns", "signal_extraction", "ratio", "cuts", "enable_legend", 'total_lumi']

    def auto_decorate(self):
        if hasattr(self, "dist") and self.dist:
            self.is_multidimensional = True if ":" in self.dist.replace("::", "") else False
        if self.xtitle is None and hasattr(self, "dist") and self.dist:
            self.xtitle = self.dist
        elif self.xtitle is None:
            self.xtitle = ""

    def merge_configs(self, other):
        """
        Merge two plot configs

        :param other: plot config to be merged in self
        :type other: PlotConfig
        :return: None
        :rtype: None
        """
        previous_choice = None
        if other is None:
            return
        for attr, val in list(other.__dict__.items()):
            if not hasattr(self, attr):
                setattr(self, attr, val)
                continue
            if getattr(self, attr) != val:
                default = '[default = {:d}]'.format(previous_choice) if previous_choice is not None else ''
                arg = "Different settings for attribute {:s} in common configs." \
                      "Please choose 1) {:s} or 2) {:s}   {:s}: ".format(attr, str(val),
                                                                         str(getattr(self, attr)),
                                                                         default)
                dec = eval(input(arg))

                if dec == "1" or (dec != '2' and previous_choice == 1):
                    setattr(self, attr, val)
                    previous_choice = 1
                elif dec == "2" or (dec != '1' and previous_choice == 2):
                    previous_choice = 2
                    continue
                else:
                    _logger.warning("Invalid choice {:s}. Take {:s}".format(str(dec), str(getattr(self, attr))))

    def get_lumi(self):
        if not isinstance(self.lumi, OrderedDict) and not isinstance(self.lumi, dict):
            return self.lumi
        if self.total_lumi is not None:
            return self.total_lumi
        if len(self.used_mc_campaigns):
            self.total_lumi = sum([self.lumi[tag] for tag in set(self.used_mc_campaigns)])
            return self.total_lumi


default_plot_config = PlotConfig(name=None)


def get_default_plot_config(hist):
    """
    Get plot config with default arguments and name according to histogram name
    :param hist: histogram object
    :type hist: THX
    :return: plot configuration
    :rtype: PlotConfig
    """
    return PlotConfig(name=hist.GetName())


def get_default_color_scheme():
    # return [ROOT.kBlack,  ROOT.kBlue-6, ROOT.kGreen+2, ROOT.kRed, ROOT.kGray, ROOT.kYellow-3, ROOT.kTeal - 2,
    # ROOT.kRed+2, ROOT.kCyan,  ROOT.kBlue, ROOT.kSpring-8]

    return [ROOT.kGray + 3,
            ROOT.kRed + 2,
            ROOT.kAzure + 4,
            ROOT.kSpring - 6,
            ROOT.kOrange - 3,
            ROOT.kCyan - 3,
            ROOT.kPink - 2,
            ROOT.kSpring - 9,
            ROOT.kMagenta - 5,
            ROOT.kOrange,
            ROOT.kCyan + 3,
            ROOT.kPink + 4,
            ROOT.kGray + 3,
            ROOT.kRed + 2,
            ROOT.kAzure + 4,
            ROOT.kSpring - 6,
            ROOT.kOrange - 3,
            ROOT.kCyan - 3,
            ROOT.kPink - 2,
            ROOT.kSpring - 9,
            ROOT.kMagenta - 5,
            ROOT.kOrange,
            ROOT.kCyan + 3,
            ROOT.kPink + 4]


def parse_mc_campaign(process_name):
    if 'mc16a' in process_name.lower():
        return 'mc16a'
    elif 'mc16c' in process_name.lower():
        return 'mc16c'
    elif 'mc16d' in process_name.lower():
        return 'mc16d'
    if 'mc16e' in process_name.lower():
        return 'mc16e'
    return None

# obsolete?
# def expand_plot_config(plot_config):
#     plot_configs = []
#     if hasattr(plot_config, "cuts_ref"):
#         if "dummy1" in plot_config.cuts_ref:
#             # for cuts in plot_config.cuts_ref.values():
#             for item in ["dummy1", "dummy2", "dummy3", "dummy4", "dummy5", "dummy6", "dummy7", "dummy8"]:
#                 if item not in plot_config.cuts_ref:
#                     continue
#                 cuts = plot_config.cuts_ref[item]
#                 tmp_config = copy(plot_config)
#                 tmp_config.cuts = cuts
#                 plot_configs.append(tmp_config)
#         else:
#             for cut_name, cut in list(plot_config.cuts_ref.items()):
#                 cuts = plot_config.cuts_ref[cut_name]
#                 tmp_config = copy(plot_config)
#                 tmp_config.cuts = plot_config.cuts + cuts
#                 tmp_config.enable_cut_ref_merge = True
#                 tmp_config.name += "_{:s}_".format(cut_name)
#                 plot_configs.append(tmp_config)
#     else:
#         for dist in plot_config.dist:
#             tmp_config = copy(plot_config)
#             tmp_config.dist = dist
#             plot_configs.append(tmp_config)
#     return plot_configs


def parse_and_build_plot_config(config_file):
    try:
        parsed_config = yl.read_yaml(config_file)
        common_plot_config = None
        if "common" in parsed_config:
            common_plot_config = PlotConfig(name="common", is_common=True, **(parsed_config["common"]))
        plot_configs = [PlotConfig(name=k, **v) for k, v in list(parsed_config.items()) if not k == "common"]
        _logger.debug("Successfully parsed {:d} plot configurations.".format(len(plot_configs)))
        return plot_configs, common_plot_config
    except Exception as e:
        raise e


def parse_and_build_process_config(process_config_files):
    """
    Parse yml file containing process definition and build ProcessConfig object
    :param process_config_files: process configuration yml files
    :type process_config_files: list
    :return: Process config
    :rtype: ProcessConfig
    """
    if process_config_files is None:
        return None
    try:
        _logger.debug("Parsing process configs")
        if not isinstance(process_config_files, list):
            parsed_process_config = yl.read_yaml(process_config_files)
            process_configs = {k: ProcessConfig(name=k, **v) for k, v in list(parsed_process_config.items())}
        else:
            parsed_process_configs = [yl.read_yaml(pcf) for pcf in process_config_files]
            process_configs = {k: ProcessConfig(name=k, **v) for parsed_config in parsed_process_configs
                               for k, v in list(parsed_config.items())}
        _logger.debug("Successfully parsed %i process items." % len(process_configs))
        return process_configs
    except Exception as e:
        raise e


def merge_plot_configs(plot_configs):
    merged_plot_config = None
    merged_common_config = None
    for plot_config, common_config in plot_configs:
        if merged_plot_config is None:
            merged_plot_config = plot_config
            merged_common_config = common_config
            continue
        merged_plot_config += plot_config
        if merged_common_config is None:
            merged_common_config = common_config
            continue
        merged_common_config.merge_configs(common_config)
    return merged_plot_config, merged_common_config


def propagate_common_config(common_config, plot_configs):
    """
    Propagate common config settings to all plot configs
    :param common_config: Common settings shared among plot configs
    :type common_config: PlotConfig
    :param plot_configs: all defined plot configs
    :type plot_configs: PlotConfig
    :return: Nothing
    :rtype: None
    """

    def integrate(plot_config, attr, value):
        if attr == "cuts":
            if plot_config.cuts is not None and value is not None:
                plot_config.cuts += value
                return
        if attr == "weight":
            if plot_config.weight is not None and not plot_config.weight.lower() == "none":
                plot_config.weight += " * {:s}".format(value)
            else:
                plot_config.weight = value
        if hasattr(plot_config, attr) and getattr(plot_config, attr) != getattr(default_plot_config, attr) and \
                attr not in PlotConfig.get_overwritable_options() or attr is None:
            return
        if hasattr(default_plot_config, attr) and value == getattr(default_plot_config, attr):
            return
        if attr == "ratio_config":
            plot_config.ratio_config = deepcopy(value)
            return
        setattr(plot_config, attr, value)

    for attr, value in list(common_config.__dict__.items()):
        for plot_config in plot_configs:
            integrate(plot_config, attr, value)


def _parse_draw_option(plot_config, process_config=None):
    draw_option = "Hist"
    if hasattr(plot_config, "draw"):
        draw_option = plot_config.draw
    if process_config and hasattr(process_config, "draw"):
        draw_option = process_config.draw
    return draw_option


def get_draw_option_as_root_str(plot_config, process_config=None):
    if plot_config.draw_option is not None:
        return plot_config.draw_option
    draw_option = _parse_draw_option(plot_config, process_config)
    if draw_option == "Marker":
        draw_option = "p"
    elif draw_option == "MarkerError":
        draw_option = "E"
    elif draw_option == "Line":
        draw_option = "l"
    elif draw_option == "hist":
        draw_option = "HIST"
    return draw_option


def transform_color(color, index=None):
    if isinstance(color, "".__class__) or isinstance(color, u"".__class__):
        offset = 0
        if "+" in color:
            color, offset = color.split("+")
        if "-" in color:
            color, offset = color.split("-")
            offset = "-" + offset
        color = getattr(ROOT, color.rstrip().replace('ROOT.', '')) + int(str(offset).replace(' ', ''))

    if isinstance(color, list):
        try:
            return transform_color(color[index])
        except IndexError:
            _logger.error("Requested {:d}th color, but only provided {:d} colors in config. "
                          "Returning black".format(index + 1, len(color)))
            return ROOT.kBlack
    return color


def get_style_setters_and_values(plot_config, process_config=None, index=None):
    """
    Parse style setter and values from draw option and config files
    :param plot_config: current plot configuration
    :type plot_config: PlotConfig
    :param process_config: process configuration
    :type process_config: ProcessConfig
    :param index: index of current object in multi-object plotting
    :type index: int
    :return: style setters (Marker, Line etc), attribute (color, size etc) and color
    :rtype: list
    """
    style_setter = None
    style_attr, color = None, None
    draw_option = _parse_draw_option(plot_config, process_config)
    if hasattr(process_config, "style"):
        style_attr = process_config.style
    if hasattr(plot_config, "styles") and index is not None:
        style_attr = plot_config.styles[index]
    if plot_config.style is not None:
        style_attr = plot_config.style
    if hasattr(process_config, "color"):
        color = transform_color(process_config.color)
    if draw_option.lower() == "hist" or re.match(r"e\d", draw_option.lower()):
        if hasattr(process_config, "format"):
            try:
                style_setter = process_config.format.capitalize()
            except AttributeError:
                _logger.error('Problem getting style from format ')
        elif style_attr:
            # TODO: needs fix
            # style_setter = 'Line'
            style_setter = "Fill"
        else:
            # style_setter = ["Line", "Marker", "Fill"]
            style_setter = ["Line"]
    elif draw_option.lower() == "marker" or draw_option.lower() == "markererror" or draw_option.lower() == 'pLX':
        style_setter = ["Marker", 'Line']
    elif draw_option.lower() == "line":
        style_setter = "Line"
    if hasattr(plot_config, "style_setter"):
        style_setter = plot_config.style_setter
    if plot_config.color is not None:
        if index is None:
            index = 0
        elif isinstance(plot_config.color, list) and index > len(plot_config.color):
            index = index % len(plot_config.color)
            style_attr = 10
        color = transform_color(plot_config.color, index)

    # else:
    #     style_attr = None
    if not isinstance(style_setter, list):
        style_setter = [style_setter]
    _logger.debug("Parsed style setter {:s} from draw option {:s}".format(str(style_setter), str(draw_option)))
    return style_setter, style_attr, color


def get_histogram_definition(plot_config, systematics='Nominal', factor_syst=''):
    """
    Create histogram defintion based on plot configuration. Dimension is parsed counting : in the distribution. If no
    distribution is provided by default a one dimension histogram will be created
    :param plot_config: plot configuration with binning and name
    :type plot_config: PlotConfig
    :param systematics: name of systematic uncertainty (default: Nominal)
    :type systematics: str
    :param factor_syst: scale factor systematic uncertainty name
    :type factor_syst:  str
    :return: histogram
    :rtype: ROOT.THXF
    """
    if plot_config.dist is not None:
        dimension = plot_config.dist.replace("::", "").count(":")
    else:
        dimension = -1
    hist = None
    hist_name = check_name('{:s}%%{:s}_{:s}%%'.format(plot_config.name, str(systematics), factor_syst))
    if dimension == 0:
        if not plot_config.logx:
            hist = ROOT.TH1F(hist_name, "", plot_config.bins, plot_config.xmin, plot_config.xmax)
        else:
            logxmin = log10(plot_config.xmin)
            logxmax = log10(plot_config.xmax)
            binwidth = old_div((logxmax - logxmin), plot_config.bins)
            xbins = []
            for i in range(0, plot_config.bins + 1):
                xbins.append(pow(10, logxmin + i * binwidth))
            hist = ROOT.TH1F(hist_name, "", plot_config.bins, array('d', xbins))
    elif dimension == 1:
        if isinstance(plot_config.xbins, list):
            hist = ROOT.TH2F(hist_name, "", len(plot_config.xbins) - 1, array("d", plot_config.xbins),
                             plot_config.ybins, plot_config.ymin, plot_config.ymax)
        else:
            if plot_config.ybins is not None:
                hist = ROOT.TH2F(hist_name, "", plot_config.xbins, plot_config.xmin, plot_config.xmax,
                                 plot_config.ybins, plot_config.ymin, plot_config.ymax)
            else:
                hist = ROOT.TProfile(hist_name, "", plot_config.xbins, plot_config.xmin, plot_config.xmax,
                                     plot_config.ymin, plot_config.ymax)
    elif dimension == 2:
        hist = ROOT.TH3F(hist_name, "", plot_config.xbins, plot_config.xmin, plot_config.xmax,
                         plot_config.ybins, plot_config.ymin, plot_config.ymax,
                         plot_config.zbins, plot_config.zmin, plot_config.zmax)
    if not hist:
        _logger.error("Unable to create histogram for plot_config %s for variable %s" % (plot_config.name,
                                                                                         plot_config.dist))
        raise InvalidInputError("Invalid plot configuration")
    hist.Sumw2()
    return hist


# todo: remove if not needed anymore
# def add_campaign_specific_merge_process(process_config, process_configs, campaign_tag):
#     new_config = deepcopy(process_config)
#     for index, sub_process in enumerate(process_config.subprocesses):
#         if 're.' not in sub_process:
#             print 'Problem, this is not covered yet - process:', process_config.name
#             #raw_input('Hit enter to acknowledge and complain on jira.')
#             continue
#         if 'mc' not in sub_process:
#             process_config.subprocesses[index] = sub_process + '([^(({:s})]$)'.format(campaign_tag)
#         elif campaign_tag not in sub_process:
#             split_info = sub_process.split(')]$')
#             process_config.subprocesses[index] = split_info[0] + '|| ' + campaign_tag + split_info[1] + ')]$)'
#
#     new_config.name += '.{:s}'.format(campaign_tag)
#     for index, sub_process in enumerate(new_config.subprocesses):
#         new_config.subprocesses[index] = sub_process + '({:s})$'.format(campaign_tag)
#     new_config.parent_process = process_config
#     process_configs[new_config.name] = new_config


def find_process_config(process, process_configs):
    """
    Searches for process config matching process name. If process name matches subprocess of mother process it adds a
    new process config to process_configs. If a MC campaign is parsed and it is a subprocess and no mother process with
    MC campaign info exists it will be created adding
    :param process_name:
    :type process_name:
    :param process_configs:
    :type process_configs:
    :return:
    :rtype:
    """

    def is_sub_process(config):
        if process.match(config.name):
            return True
        if not hasattr(config, 'subprocesses'):
            return False
        if process.matches_any(config.subprocesses) is not None:
            return True
        return False

    if not isinstance(process, Process):
        return find_process_config_str(process, process_configs)
    if process_configs is None or process is None:
        return None
    match = process.matches_any(list(process_configs.keys()))
    if match is not None:
        return process_configs[match]
    matched_process_cfg = [pc for pc in list(process_configs.values()) if is_sub_process(pc)]
    if len(matched_process_cfg) != 1:
        if len(matched_process_cfg) > 0:
            print('SOMEHOW matched to multiple configs')
        return None
    return matched_process_cfg[0]


def find_process_config_str(process_name, process_configs):
    """
    Searches for process config matching process name. If process name matches subprocess of mother process it adds a
    new process config to process_configs. If a MC campaign is parsed and it is a subprocess and no mother process with
    MC campaign info exists it will be created adding
    :param process_name:
    :type process_name:
    :param process_configs:
    :type process_configs:
    :return:
    :rtype:
    """

    def is_sub_process(config):
        if process_name == config.name:
            return True
        if not hasattr(config, 'subprocesses'):
            return False
        if process_name in config.subprocesses:
            return True
        if any(re.match(sub_process.replace("re.", ""), process_name) for sub_process in config.subprocesses):
            return True
        return False

    if process_configs is None or process_name is None:
        return None
    if process_name in process_configs:
        return process_configs[process_name]
    matched_process_cfg = [pc for pc in list(process_configs.values()) if is_sub_process(pc)]
    if len(matched_process_cfg) != 1:
        if len(matched_process_cfg) > 0:
            print('SOMEHOW matched to multiple configs')
        return None
    return matched_process_cfg[0]
