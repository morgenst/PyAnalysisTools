__author__ = 'marcusmorgenstern'
__mail__ = ''

from Logger import Logger

class InvalidInputError(ValueError):
    pass

_logger=Logger.retrieve_logger()