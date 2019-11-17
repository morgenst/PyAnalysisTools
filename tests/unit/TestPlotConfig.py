import unittest
import math
from PyAnalysisTools.PlottingUtils.PlotConfig import PlotConfig


class TestObjectHandle(unittest.TestCase):
    def setUp(self):
        pass

    def test_math_range(self):
        pc = PlotConfig(ymin='math.pi')
        self.assertEqual(math.pi, pc.ymin)
