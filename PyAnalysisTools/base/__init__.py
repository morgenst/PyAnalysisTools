__author__ = 'marcusmorgenstern'
__mail__ = ''

import ROOT
from Logger import Logger

class InvalidInputError(ValueError):
    pass

_logger=Logger().retrieve_logger()

"""
Setting memory policy to avoid graphics objects getting deleted once drawn to canvas and canvas passing around.
"""
ROOT.SetMemoryPolicy(ROOT.kMemoryStrict)
