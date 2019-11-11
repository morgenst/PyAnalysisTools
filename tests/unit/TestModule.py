import unittest
import os
from PyAnalysisTools.base.Modules import load_modules


class TestModule(unittest.TestCase):
    def setUp(self):
        pass

    def test_module_load(self):
        modules = load_modules(os.path.join(os.path.dirname(__file__), 'fixtures/module_config.yml'), self)
        self.assertEqual(1, len(modules))
