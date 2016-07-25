import sys
import yaml
from . import _logger


class YAMLLoader(object):
    def __init__(self, **kwargs):
        kwargs.setdefault('log_level', 'warning')
        for k, v in kwargs.iteritems():
            setattr(self, k, v)

    @staticmethod
    def read_yaml(file_name):
        try:
            _logger.debug("Try to open yaml file %s" % file_name)
            config_file = open(file_name)
            _logger.debug("Try to load content of %s" % file_name)
            config = yaml.load(config_file)
            config_file.close()
            return config
        except IOError as e:
            _logger.error("Could not find or open yaml file %s" % file_name)
            _logger.error(e.strerror)
            raise e
        except yaml.YAMLError as e:
            _logger.error("Failed to load yaml file %s" % file_name)
            raise e
        except Exception as e:
            _logger.error("Unexpected error for %s" % file_name)
            raise e


class YAMLDumper(object):
    def __init__(self, **kwargs):
        kwargs.setdefault('log_level', 'warning')
        for k, v in kwargs.iteritems():
            setattr(self, k, v)

    @staticmethod
    def dump_yaml(data, file_name):
        try:
            _logger.debug("Try to open file %s" % file_name)
            out_file = open(file_name, "w+")
            _logger.debug("Try to dump data to file %s" % file_name)
            yaml.dump(data, out_file)
            out_file.close()
        except Exception as e:
            _logger.error("Failed to dump data to file %s" % file_name)
            raise e