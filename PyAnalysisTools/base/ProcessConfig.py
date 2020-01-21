from __future__ import unicode_literals

from builtins import str
from builtins import map
from builtins import object
from future.utils import python_2_unicode_compatible
import re
from PyAnalysisTools.base import _logger
from PyAnalysisTools.base.YAMLHandle import YAMLLoader as yl

data_streams = ['physics_Late', 'physics_Main']


@python_2_unicode_compatible
class Process(object):
    """
    Class defining a physics process
    """

    def __init__(self, file_name, dataset_info, process_name=None, tags=[], cut=None, weight=None):
        """
        Constructor
        :param file_name: name of input file
        :type file_name: str
        :param dataset_info: config containing information on all defined datasets
        :type dataset_info: dict
        """
        self.dataset_info = dataset_info
        self.file_name = file_name
        self.stream = None
        self.is_mc = False
        self.is_data = False
        self.dsid = None
        self.mc_campaign = None
        self.cut = cut
        self.tags = re.compile('-?|'.join(map(re.escape, ['hist', 'ntuple'] + tags + [''])))
        if file_name is not None:
            self.base_name = self.tags.sub('', file_name).lstrip('-').replace('.root', '')
        else:
            self.base_name = None
        self.year = None
        self.period = None
        self.process_name = process_name
        self.weight = weight
        if file_name is not None:
            self.parse_file_name(self.base_name.split('/')[-1])
        if self.cut is not None:
            self.process_name += self.cut

    def __str__(self):
        """
        Overloaded str operator. Get's called if object is printed
        :return: formatted string with name and attributes
        :rtype: str
        """
        obj_str = str(self.process_name)
        obj_str += ' parsed from file name {:s}'.format(str(self.file_name))
        return obj_str

    def __unicode__(self):
        """
        Overloaded unicode str operator. Get's called if object is printed
        :return: formatted string with name and attributes
        :rtype: str
        """
        obj_str = str(self.process_name)
        obj_str += ' parsed from file name {:s}'.format(self.file_name)
        return obj_str

    def __format__(self, format_spec):
        """
        Overloaded format operated called when formatted string output is requested.
        :param format_spec:
        :return: unicode str
        """
        return self.__unicode__()

    def __eq__(self, other):
        """
        Comparison operator
        :param other: plot config object to compare to
        :type other: PlotConfig
        :return: True/False
        :rtype: boolean
        """
        if isinstance(self, other.__class__):
            for k, v in list(self.__dict__.items()):
                if k not in other.__dict__:
                    return False
                if k in ['base_name', 'dataset_info', 'file_name', 'tags']:
                    continue
                if self.__dict__[k] != other.__dict__[k]:
                    return False
        else:
            return False
        return True

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
        return hash(self.process_name)

    def parse_file_name(self, file_name):
        """
        Reads process information from given file name
        :param file_name: file name
        :type file_name: str
        :return: nothing
        :rtype: None
        """
        if 'data' in file_name:
            self.set_data_name(file_name)
        elif re.match(r'\d{6}', file_name):
            self.set_mc_name(file_name)
        else:
            _logger.debug("No dedicated parsing found. Assume MC and run simplified")
            self.set_mc_name(file_name)

    def set_data_name(self, file_name):
        """
        Parser for data file analysing year and potential period
        :param file_name: file name
        :type file_name: str
        :return: nothing
        :rtype: None
        """
        self.is_data = True
        if 'period' in file_name:
            self.year, _, self.period = file_name.split("_")[0:3]
            self.process_name = ".".join([self.year, self.period])
        elif 'allyear' in file_name.lower():
            self.year = re.search('[0-9]{2}', re.search(r'data[0-9]{2}', file_name).group()).group()
            self.process_name = 'data{:s}_allYear'.format(self.year)
        elif re.search(r'00\d{6}', file_name):
            self.year = re.search('[0-9]{2}', re.search(r'data[0-9]{2}', file_name).group()).group()
            self.process_name = 'data{:s}_{:s}'.format(self.year, re.search(r'00\d{6}', file_name).group())

    def set_mc_name(self, file_name):
        """
        Parser for MonteCarlo processes
        :param file_name: file name
        :type file_name: str
        :return: nothing
        :rtype: None
        """
        self.is_mc = True
        try:
            self.dsid = re.search(r'\d{6,}', file_name).group(0)
        except AttributeError:
            pass
        if re.search(r'\d{6,}', file_name):
            self.parse_from_dsid()
        else:
            self.process_name = file_name
        self.parse_mc_campaign(file_name)

    def parse_from_dsid(self):
        """
        Read information from dataset info 'DB' given the dataset id (dsid) - for MC only
        :return: nothing
        :rtype: None
        """
        if self.dataset_info is None:
            self.process_name = self.dsid
            return
        try:
            tmp = [l for l in list(self.dataset_info.values()) if l.dsid == int(self.dsid)]
        except ValueError:
            _logger.error("Could not find {:d}".format(self.dsid))
        if len(tmp) == 1:
            self.process_name = tmp[0].process_name

    def parse_mc_campaign(self, file_name):
        """
        Parse the MC production campaign from file name
        :param file_name: file name
        :type file_name:str
        :return: nothing
        :rtype: None
        """
        if 'mc16a' in file_name.lower():
            self.mc_campaign = 'mc16a'
        if 'mc16c' in file_name.lower():
            self.mc_campaign = 'mc16c'
        if 'mc16d' in file_name.lower():
            self.mc_campaign = 'mc16d'
        if 'mc16e' in file_name.lower():
            self.mc_campaign = 'mc16e'

    def matches_any(self, process_names):
        """
        Check if this process matches any name provided in process_names
        :param process_names: list of process names to be checked against
        :type process_names: list<str>
        :return: matched process name if succeeded; otherwise None
        :rtype: str or None
        """
        for process_name in process_names:
            if self.match(process_name):
                return process_name

    def match(self, process_name):
        if self.process_name is None:
            return False
        if 're.' in process_name:
            return re.match(process_name.replace('re.', ''), self.process_name)
        return process_name == self.process_name


class ProcessConfig(object):
    def __init__(self, **kwargs):
        kwargs.setdefault('parent_process', None)
        kwargs.setdefault('scale_factor', None)
        kwargs.setdefault('regions_only', None)
        for k, v in list(kwargs.items()):
            setattr(self, k.lower(), v)
        self.is_data, self.is_mc = self.transform_type()

    def __str__(self):
        """
        Overloaded str operator. Get's called if object is printed
        :return: formatted string with name and attributes
        :rtype: str
        """
        obj_str = "Process config: {:s} \n".format(self.name)
        for attribute, value in list(self.__dict__.items()):
            obj_str += '{}={} \n'.format(attribute, value)
        return obj_str

    def __repr__(self):
        """
        Overloads representation operator. Get's called e.g. if list of objects are printed
        :return: formatted string with name and attributes
        :rtype: str
        """
        return self.__str__() + '\n'

    def transform_type(self):
        """
        Initialise MC or data type
        :return: pair of TRUE/FALSE for is_data and is_mc
        :rtype: (bool, bool)
        """
        if "data" in self.type.lower():
            return True, False
        else:
            return False, True

    def retrieve_subprocess_config(self):
        """
        Retrieve all sub-process configurations
        :return: sub-process configs
        :rtype: dict
        """
        tmp = {}
        if not hasattr(self, "subprocesses"):
            return tmp
        for sub_process in self.subprocesses:
            tmp[sub_process] = ProcessConfig(
                **dict((k, v) for (k, v) in list(self.__dict__.items()) if not k == "subprocesses"))
        return tmp

    def add_subprocess(self, subprocess_name):
        """
        Add a sub-process to process and propagate config to it
        :param subprocess_name: name of sub-process
        :type subprocess_name: str
        :return:
        :rtype:
        """
        self.subprocesses.append(subprocess_name)
        pc = ProcessConfig(**dict((k, v) for (k, v) in list(self.__dict__.items()) if not k == "subprocesses"))
        pc.parent_process = self.name
        return pc


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