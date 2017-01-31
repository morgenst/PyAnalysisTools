import unittest
import logging
from PyAnalysisTools.base import Logger


class TestLogger(unittest.TestCase):
    def setUp(self):
        self.logger = Logger()

    def test_ctor_log_level(self):
        self.assertEqual(self.logger.logger.getEffectiveLevel(), logging.WARNING)

    def test_set_log_level(self):
        Logger.set_log_level(self.logger.logger, "debug")
        self.assertEqual(self.logger.logger.getEffectiveLevel(), logging.DEBUG)

    def test_set_log_invalid_level(self):
        Logger.set_log_level(self.logger.logger, "foo")
        self.assertEqual(self.logger.logger.getEffectiveLevel(), logging.WARNING)

    def test_get_logger(self):
        tmp_logger = self.logger.retrieve_logger()
        self.assertEqual(tmp_logger, self.logger.logger)

    def test_get_help_message(self):
        msg = "Log level. Options: [info, warning, error, debug]"
        self.assertEqual(Logger.get_help_msg(), msg)

