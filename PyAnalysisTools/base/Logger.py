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
        self._set_log_level(level)
        self.logger.propagate = 0

    def retrieve_logger(self):
        return self.logger

    def _set_log_level(self, level):
        try:
            self.logger.setLevel(getattr(logging, level.upper()))
        except AttributeError:
            self.logger.setLevel(logging.WARNING)