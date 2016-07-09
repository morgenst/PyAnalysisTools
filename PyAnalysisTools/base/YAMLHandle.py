import sys
import yaml
from Logger import Logger


class YAMLLoader(object):
    def __init__(self, **kwargs):
        kwargs.setdefault('log_level', 'warning')
        for k, v in kwargs.iteritems():
            setattr(self, k, v)
        self._logger = Logger(self.__class__.__name__, self.log_level).retrieve_logger()

    def read_yaml(self, file_name):
        try:
            self._logger.debug("Try to open yaml file %s" % file_name)
            config_file = open(file_name)
            self._logger.debug("Try to load content of %s" % file_name)
            config = yaml.load(config_file)
            config_file.close()
            return config
        except IOError as e:
            self._logger.error("Could not find or open yaml file %s" % file_name)
            self._logger.error(e.strerror)
            raise e
        except yaml.YAMLError as e:
            self._logger.error("Failed to load yaml file %s" % file_name)
            raise e
        except Exception as e:
            self._logger.error("Unexpected error for %s" % file_name)
            raise e


class YAMLDumper(object):
    def __init__(self, **kwargs):
        kwargs.setdefault('log_level', 'warning')
        for k, v in kwargs.iteritems():
            setattr(self, k, v)
        self._logger = Logger(self.__class__.__name__, self.log_level).retrieve_logger()

    def dump_yaml(self, data, file_name):
        try:
            self._logger.debug("Try to open file %s" % file_name)
            out_file = open(file_name, "w+")
            self._logger.debug("Try to dump data to file %s" % file_name)
            yaml.dump(data, out_file)
            out_file.close()
        except Exception as e:
            self._logger("Failed to dump data to file %s" % file_name)
            raise e