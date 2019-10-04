from __future__ import unicode_literals
from __future__ import print_function

from collections import OrderedDict
from copy import deepcopy

from builtins import input
from builtins import object
try:
    import oyaml as yaml
except ImportError:
    print("yaml has been replaced by oyaml to provide support for ordered dictionaries read from configuration")
    print("Please install via: \033[91m pip install oyaml --user.\033[0m")
    _ = input("Acknowledge by hitting enter (running with yaml for now. Note this might cause crashes)")
    #import yaml
    pass
from . import _logger


class YAMLLoader(object):
    """
    Class to load data from yaml files using oyaml module for ordered dicts
    """
    def __init__(self, **kwargs):
        kwargs.setdefault('log_level', 'warning')
        for k, v in kwargs.items():
            setattr(self, k, v)

    def construct_yaml_str(self, node):
        # Override the default string handling function
        # to always return unicode objects
        return self.construct_scalar(node)
    global default_ctor
    try:
        default_ctor = deepcopy(yaml.pyyaml.Loader.__dict__['yaml_constructors'])
    except KeyError:
        pass
    yaml.pyyaml.Loader.add_constructor(u'tag:yaml.org,2002:str', construct_yaml_str)

    @staticmethod
    def read_yaml(file_name, accept_none=False):
        """
        Read data from given file and return it
        :param file_name: name of yaml file
        :type file_name: str
        :param accept_none: switch to deal with no file name provided (returns None)
        :type accept_none: bool
        :return: loaded data
        :rtype: any
        """
        def convert_to_ordered_dict(item):
            """
            Convert dict to OrderedDict. Required because since python 3.7 dicts are automatically ordered, but in
            several places it is checked if an object is an instance of OrderedDict. Parses recursively through dict
            and lists.
            :param item: current entry
            :return: converted entry
            """
            if isinstance(item, dict):
                item = OrderedDict(item)
                for k, v in item.items():
                    item[k] = convert_to_ordered_dict(v)
            if isinstance(item, list):
                item = [convert_to_ordered_dict(i) for i in item]
            return item

        if accept_none and file_name is None:
            return None
        try:
            try:
                with open(file_name, 'r') as config_file:
                    config = yaml.load(config_file, Loader=yaml.pyyaml.Loader)
            except TypeError:
                #workaround for existing yaml files written with py2 loader
                with open(file_name, 'r') as config_file:
                    setattr(yaml.pyyaml.Loader, 'yaml_constructors', default_ctor)
                    config = yaml.load(config_file, Loader=yaml.pyyaml.Loader)
            config = convert_to_ordered_dict(config)
            return config
        except IOError as e:
            _logger.error("Could not find or open yaml file %s" % file_name)
            _logger.error(e.strerror)
            raise e
        except yaml.YAMLError as e:
            _logger.error("Failed to load yaml file %s" % file_name)
            raise e


class YAMLDumper(object):
    """
    Wrapper to write data to yaml file
    """
    def __init__(self, **kwargs):
        kwargs.setdefault('log_level', 'warning')
        for k, v in kwargs.items():
            setattr(self, k, v)

    @staticmethod
    def dump_yaml(data, file_name, **kwargs):
        """
        Converts and writes data to yaml file
        :param data: data to be stored
        :type data: ant
        :param file_name: output file name
        :type file_name: str
        :param kwargs: optional arguments
        :type kwargs: dict
        :return: nothing
        :rtype: None
        """
        try:
            _logger.debug("Try to open file {:s}".format(file_name))
            out_file = open(file_name, "w+")
            _logger.debug("Try to dump data to file {:s}".format(file_name))
            yaml.dump(data, out_file, **kwargs)
            out_file.close()
        except Exception as e:
            _logger.error("Failed to dump data to file {:s}".format(file_name))
            raise e
