import os
import unittest

import ROOT
from PyAnalysisTools.PlottingUtils import PlotableObject as po

cwd = os.path.dirname(__file__)
ROOT.gROOT.SetBatch(True)


class TestPlotableObject(unittest.TestCase):
    def test_ctor(self):
        obj = po.PlotableObject()
        self.assertIsNone(obj.plot_object)
        self.assertTrue(obj.is_ref)
        self.assertEqual(-1, obj.ref_id)
        self.assertEqual('', obj.label)
        self.assertIsNone(obj.cuts)
        self.assertIsNone(obj.process)
        self.assertEqual('Marker', obj.draw_option)
        self.assertEqual('Marker', obj.draw)
        self.assertEqual(1, obj.marker_color)
        self.assertEqual(1, obj.marker_size)
        self.assertEqual(1, obj.marker_style)
        self.assertEqual(1, obj.line_color)
        self.assertEqual(1, obj.line_width)
        self.assertEqual(1, obj.line_style)
        self.assertEqual(0, obj.fill_color)
        self.assertEqual(0, obj.fill_style)

    def tests_palettes(self):
        color_palette = [ROOT.kGray + 3, ROOT.kRed + 2, ROOT.kAzure + 4, ROOT.kSpring - 6, ROOT.kOrange - 3,
                         ROOT.kCyan - 3, ROOT.kPink - 2, ROOT.kSpring - 9, ROOT.kMagenta - 5]
        marker_style_palette_filled = [21, 20, 22, 23, 33, 34, 29, 2]
        marker_style_palette_empty = [25, 24, 26, 32, 27, 28, 30, 5]
        line_style_palette_homogen = [1, 9, 7, 2, 3]
        line_style_palette_heterogen = [10, 5, 4, 8, 6]
        fill_style_palette_left = [3305, 3315, 3325, 3335, 3345, 3365, 3375, 3385]
        fill_style_palette_right = [3359, 3351, 3352, 3353, 3354, 3356, 3357, 3358]

        self.assertEqual(color_palette, po.color_palette)
        self.assertEqual(marker_style_palette_filled, po.marker_style_palette_filled)
        self.assertEqual(marker_style_palette_empty, po.marker_style_palette_empty)
        self.assertEqual(line_style_palette_homogen, po.line_style_palette_homogen)
        self.assertEqual(line_style_palette_heterogen, po.line_style_palette_heterogen)
        self.assertEqual(fill_style_palette_left, po.fill_style_palette_left)
        self.assertEqual(fill_style_palette_right, po.fill_style_palette_right)
