import re

data_streams = ['physics_Late', 'physics_Main']


class Process(object):
    def __init__(self, file_name, dataset_info):
        self.dataset_info = dataset_info
        self.stream = None
        self.is_mc = False
        self.is_data = False
        self.dsid = None
        self.mc_campaign = None
        self.base_name = file_name.replace('hist-', '').replace('ntuple-', '').replace('.root', '')
        self.year = None
        self.period = None
        self.process_name = None
        self.parse_file_name(self.base_name.split('/')[-1])

    def __str__(self):
        """
        Overloaded str operator. Get's called if object is printed
        :return: formatted string with name and attributes
        :rtype: str
        """
        obj_str = str(self.process_name)
        obj_str += ' parsed from file name {:s}'.format(self.base_name)
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
                if k in ['base_name', 'dataset_info']:
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
        if 'data' in file_name:
            self.set_data_name(file_name)
        elif re.match(r'\d{6}', file_name):
            self.set_mc_name(file_name)
        else:
            print "No dedicated parsing found. Assume MC and run simplified "
            self.set_mc_name(file_name)

    def set_data_name(self, file_name):
        self.is_data = True
        if 'period' in file_name:
            self.year, _, self.period = file_name.split("_")[0:3]
            self.process_name = ".".join([self.year, self.period])

    def set_mc_name(self, file_name):
        self.is_mc = True
        if re.match('\d{6}', file_name):
            self.parse_from_dsid(re.match('\d{6}', file_name).group(0))
        else:
            self.process_name = file_name
        self.parse_mc_campaign(file_name)

    def parse_from_dsid(self, dsid):
        if self.dataset_info is None:
            return
        try:
            tmp = filter(lambda l: l.dsid == int(dsid), self.dataset_info.values())
        except ValueError:
            print 'Could not find ', dsid
        if len(tmp) == 1:
            self.process_name = tmp[0].process_name

    def parse_mc_campaign(self, file_name):
        if 'mc16a' in file_name.lower():
            self.mc_campaign = 'mc16a'
        if 'mc16c' in file_name.lower():
            self.mc_campaign = 'mc16c'
        if 'mc16d' in file_name.lower():
            self.mc_campaign = 'mc16d'
        if 'mc16e' in file_name.lower():
            self.mc_campaign = 'mc16e'

    def matches_any(self, process_names):
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
        self.transform_type()

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
        if "data" in self.type.lower():
            self.is_data = True
            self.is_mc = False
        else:
            self.is_data = False
            self.is_mc = True

    def retrieve_subprocess_config(self):
        tmp = {}
        if not hasattr(self, "subprocesses"):
            return tmp
        for sub_process in self.subprocesses:
            tmp[sub_process] = ProcessConfig(**dict((k, v) for (k, v) in self.__dict__.iteritems() if not k == "subprocesses"))
        return tmp

    def add_subprocess(self, subprocess_name):
        self.subprocesses.append(subprocess_name)
        pc = ProcessConfig(**dict((k, v) for (k, v) in self.__dict__.iteritems() if not k == "subprocesses"))
        pc.parent_process = self.name
        return pc