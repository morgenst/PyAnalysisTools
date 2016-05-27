__author__ = 'marcusmorgenstern'
__mail__ = ''

import logging

class Logger():
    def __init__(self, name="base_logger", level="warning"):
        logger = logging.getLogger(name)
        hdl = logging.StreamHandler()
        form = logging.Formatter('[%(funcName)s at %(lineno)s] %(levelname)s: %(message)s')
        hdl.setFormatter(form)
        logger.addHandler(hdl)
        logger.setLevel(eval('logging.' + level))
        logger.propagate = 0

    def retrieve_logger(self):
        return self.logger