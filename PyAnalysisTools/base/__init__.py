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


def add_process_args(parser):
    """
    add default arguments for process configurations
    :param parser: argument parser
    :type parser: argparse.ArgumentParser
    :return: nothing
    :rtype: None
    """

    parser.add_argument("--lumi", "-l", type=float, default=None, help="Luminosity in fb^-1")
    parser.add_argument("--dataset_config", "-xcf", type=str, default=None, help="dataset config file")
    parser.add_argument("--process_configs", "-prcf", nargs='+', default=None, help="process definition files")


def add_output_args(parser):
    """
    add default arguments for output
    :param parser: argument parser
    :type parser: argparse.ArgumentParser
    :return: nothing
    :rtype: None
    """

    parser.add_argument("--output_dir", "-o", default=None, help="output directory")


def add_input_args(parser):
    """
    add default arguments for input
    :param parser: argument parser
    :type parser: argparse.ArgumentParser
    :return: nothing
    :rtype: None
    """

    parser.add_argument("input_file_list", nargs="+", type=str, help="input file list")
    parser.add_argument('--tree_name', '-tn', default=None, help="tree name (required for extended CA")


def add_selection_args(parser):
    """
    add default arguments for selection
    :param parser: argument parser
    :type parser: argparse.ArgumentParser
    :return: nothing
    :rtype: None
    """

    parser.add_argument('--selection_config', '-sc', default=None, help="config for additional cuts")


def add_friend_args(parser):
    """
    add default arguments for friends to be linked
    :param parser: argument parser
    :type parser: argparse.ArgumentParser
    :return: nothing
    :rtype: None
    """
    parser.add_argument('--friend_directory', '-fd', default=None, help="directory containing friend tree files")
    parser.add_argument('--friend_tree_names', '-ftn', nargs="+", default=None, help="friend tree name")
    parser.add_argument('--friend_file_pattern', '-ffp', nargs="+", help="file name patterns of friends")


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
