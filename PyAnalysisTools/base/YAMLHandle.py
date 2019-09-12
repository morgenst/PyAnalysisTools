from __future__ import print_function
try:
    import oyaml as yaml
except ImportError:
    print("yaml has been replaced by oyaml to provide support for ordered dictionaries read from configuration")
    print("Please install via: \033[91m pip install oyaml --user.\033[0m")
    _ = raw_input("Acknowledge by hitting enter (running with yaml for now. Note this might cause crashes)")
    import yaml
from . import _logger


class YAMLLoader(object):
    def __init__(self, **kwargs):
        kwargs.setdefault('log_level', 'warning')
        for k, v in kwargs.iteritems():
            setattr(self, k, v)

    @staticmethod
    def read_yaml(file_name, accept_none=False):
        if accept_none and file_name is None:
            return None
        try:
            with open(file_name, 'r') as config_file:
                config = yaml.load(config_file, Loader=yaml.pyyaml.Loader)
                return config
        except IOError as e:
            _logger.error("Could not find or open yaml file %s" % file_name)
            _logger.error(e.strerror)
            raise e
        except yaml.YAMLError as e:
            _logger.error("Failed to load yaml file %s" % file_name)
            raise e


class YAMLDumper(object):
    def __init__(self, **kwargs):
        kwargs.setdefault('log_level', 'warning')
        for k, v in kwargs.iteritems():
            setattr(self, k, v)

    @staticmethod
    def dump_yaml(data, file_name, **kwargs):
        try:
            _logger.debug("Try to open file %s" % file_name)
            out_file = open(file_name, "w+")
            _logger.debug("Try to dump data to file %s" % file_name)
            yaml.dump(data, out_file, **kwargs)
            out_file.close()
        except Exception as e:
            _logger.error("Failed to dump data to file %s" % file_name)
            raise e
