import unittest
import logging
from PyAnalysisTools.base.Logger import Logger


class TestLogger(unittest.TestCase):
    def setUp(self):
        pass

    def test_retrieve_logger(self):
        logger = Logger().retrieve_logger()
        self.assertIsInstance(logger, logging.Logger)

    def test_set_log_level_ctor(self):
        logger = Logger(level="debug").retrieve_logger()
        self.assertEqual(logger.getEffectiveLevel(), 10)

    def test_invalid_log_level_ctor(self):
        logger = Logger(level="InvalidLevel").retrieve_logger()
        self.assertEqual(logger.getEffectiveLevel(), 30)

    def test_get_help_msg(self):
        self.assertEqual('Log level. Options: [info, warning, error, debug]', Logger.get_help_msg())
