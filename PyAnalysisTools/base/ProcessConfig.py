import re
from PyAnalysisTools.base import _logger

data_streams = ['physics_Late', 'physics_Main']


class Process(object):
    """
    Class defining a physics process
    """

    def __init__(self, file_name, dataset_info, process_name=None, tags=[]):
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
        self.tags = re.compile('-?|'.join(map(re.escape, ['hist', 'ntuple'] + tags + [''])))
        if file_name is not None:
            self.base_name = self.tags.sub('', file_name).lstrip('-').replace('.root', '')
        else:
            self.base_name = None
        self.year = None
        self.period = None
        self.process_name = process_name
        if file_name is not None:
            self.parse_file_name(self.base_name.split('/')[-1])

    def __str__(self):
        """
        Overloaded str operator. Get's called if object is printed
        :return: formatted string with name and attributes
        :rtype: str
        """
        obj_str = str(self.process_name)
        obj_str += ' parsed from file name {:s}'.format(self.file_name)
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
            for k, v in self.__dict__.iteritems():
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
            self.dsid = re.search('\d{6,}', file_name).group(0)
        except AttributeError:
            pass
        if re.search('\d{6,}', file_name):
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
            tmp = filter(lambda l: l.dsid == int(self.dsid), self.dataset_info.values())
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
        for k, v in kwargs.iteritems():
            setattr(self, k.lower(), v)
        self.is_data, self.is_mc = self.transform_type()

    def __str__(self):
        """
        Overloaded str operator. Get's called if object is printed
        :return: formatted string with name and attributes
        :rtype: str
        """
        obj_str = "Process config: {:s} \n".format(self.name)
        for attribute, value in self.__dict__.items():
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
                **dict((k, v) for (k, v) in self.__dict__.iteritems() if not k == "subprocesses"))
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
        pc = ProcessConfig(**dict((k, v) for (k, v) in self.__dict__.iteritems() if not k == "subprocesses"))
        pc.parent_process = self.name
        return pc
