__author__ = 'marcusmorgenstern'
__mail__ = ''

import logging


class Logger(object):
    def __init__(self, name="base_logger", level="warning"):
        self.logger = logging.getLogger(name)
        hdl = logging.StreamHandler()
        form = logging.Formatter('[%(funcName)s at %(lineno)s] %(levelname)s: %(message)s')
        hdl.setFormatter(form)
        self.logger.addHandler(hdl)
        self.__class__.set_log_level(self.logger, level)
        self.logger.propagate = 0

    def retrieve_logger(self):
        return self.logger

    @staticmethod
    def set_log_level(logger, level):
        try:
            logger.setLevel(getattr(logging, level.upper()))
        except AttributeError:
            logger.setLevel(logging.WARNING)

    @staticmethod
    def get_help_msg():
        return "Log level. Options: [info, warning, error, debug]"
