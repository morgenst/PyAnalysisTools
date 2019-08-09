import logging


class Logger(object):
    """
    Internal class to configure a logger instance of pythons native logging package
    """
    def __init__(self, name="base_logger", level="warning"):
        self.logger = logging.getLogger(name)
        hdl = logging.StreamHandler()
        form = logging.Formatter('[%(funcName)s at %(lineno)s] %(levelname)s: %(message)s')
        hdl.setFormatter(form)
        self.logger.addHandler(hdl)
        self.__class__.set_log_level(self.logger, level)
        self.logger.propagate = 0

    def retrieve_logger(self):
        """
        Get logger instance
        :return: logger instance
        :rtype: logging.logger
        """
        return self.logger

    @staticmethod
    def set_log_level(logger, level):
        """
        Set verbosity level of logger
        :param logger: logger instance
        :type logger: logging.logger
        :param level: verbosity level (info, debug, error, warning)
        :type level: str
        :return: nothing
        :rtype: NNone
        """
        try:
            logger.setLevel(getattr(logging, level.upper()))
        except AttributeError:
            logger.setLevel(logging.WARNING)

    @staticmethod
    def get_help_msg():
        """
        Get helper message for argparser
        :return: help message
        :rtype: str
        """
        return "Log level. Options: [info, warning, error, debug]"
