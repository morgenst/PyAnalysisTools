import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
from Logger import Logger
import argparse


class InvalidInputError(ValueError):
    pass

_logger = Logger().retrieve_logger()

"""
Setting memory policy to avoid graphics objects getting deleted once drawn to canvas and canvas passing around.
"""
ROOT.SetMemoryPolicy(ROOT.kMemoryStrict)


def get_default_argparser(description=''):
    """
    Build default argument parser and add common args
    :param description: description printed when called with --help
    :type description: str
    :return: parser
    :rtype: argparse.ArgumentParser
    """

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--log_level', '-log', default='info', help='log level')
    return parser


def default_init(parser):
    """
    default initialisation of steering scripts and parsing of arguments
    :param parser: argument parser
    :type parser: argparse.ArgumentParser
    :return: parsed arguments
    :rtype: Namespace
    """

    args = parser.parse_args()
    Logger.set_log_level(_logger, args.log_level)
    return args
