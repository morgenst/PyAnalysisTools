__author__ = 'marcusmorgenstern'
__mail__ = 'marcus.matthias.morgenstern@cern.ch'

import unittest

import ROOT
#from ROOTUtils.FileHandle import FileHandle
from mock import MagicMock
import PyAnalysisTools.ROOTUtils.ObjectHandle as OH


class TestObjectHandle(unittest.TestCase):
    def setUp(self):
        self.filled_canvas = ROOT.TCanvas('c', 'c')
        self.h = ROOT.TH1F('h', 'h', 100, -1., 1.)
        self.h.FillRandom('gaus', 100)
        self.filled_canvas.cd()
        self.h.Draw()

    # def testGetObjectFromCanvas(self):
    #     obj = OH.get_objects_from_canvas(self.filled_canvas)
    #     self.assertTrue(len(obj) > 0)
    #     self.assertEqual(obj[0], self.h)
    #
    # def testGetObjectFromEmptyCanvas(self):
    #     ROOT.gROOT.SetBatch(True)
    #     c = ROOT.TCanvas("c", "c", 800, 600)
    #     obj = OH.get_objects_from_canvas(c)
    #     self.assertEqual(len(obj), 0)
    #
    # def test_get_object_from_canvas_by_ype_th1_success(self):
    #     obj = OH.get_objects_from_canvas_by_type(self.filled_canvas, "TH1")
    #     self.assertTrue(len(obj) > 0)
    #     self.assertEqual(obj[0], self.h)
    #
    # def test_get_object_from_canvas_by_ype_th1_wrong_type(self):
    #     obj = OH.get_objects_from_canvas_by_type(self.filled_canvas, "TH2")
    #     self.assertTrue(len(obj) == 0)
    #
    # def test_get_object_from_canvas_by_ype_th1_list_success(self):
    #     obj = OH.get_objects_from_canvas_by_type(self.filled_canvas, ["TH1"])
    #     self.assertTrue(len(obj) > 0)
    #     self.assertEqual(obj[0], self.h)
    #
    # def test_get_object_from_canvas_by_ype_th1_list_wrong_type(self):
    #     obj = OH.get_objects_from_canvas_by_type(self.filled_canvas, ["TH2"])
    #     self.assertTrue(len(obj) == 0)
    #
    # def test_get_object_from_canvas_by_ype_th1_success(self):
    #     obj = OH.get_objects_from_canvas_by_name(self.filled_canvas, "h")
    #     self.assertTrue(len(obj) > 0)
    #     self.assertEqual(obj[0], self.h)
    #
    # def test_get_object_from_canvas_by_ype_th1_wrong_name(self):
    #     self.assertIsNone(OH.get_objects_from_canvas_by_name(self.filled_canvas, "h2"))
    #
    # def test_get_object_from_canvas_by_ype_th1_list_success(self):
    #     obj = OH.get_objects_from_canvas_by_name(self.filled_canvas, ["h"])
    #     self.assertTrue(len(obj) > 0)
    #     self.assertEqual(obj[0], self.h)
    #
    # def test_get_object_from_canvas_by_ype_th1_list_wrong_name(self):
    #     self.assertIsNone(OH.get_objects_from_canvas_by_name(self.filled_canvas, ["h2"]))
    #
    # def test_find_branches_matching_pattern_none_found(self):
    #     tree = MagicMock()
    #     tree.get_list_of_branches = ['foo']
    #     self.assertEquals(OH.find_branches_matching_pattern(tree, 'bar'), [])

    def test_find_branches_matching_pattern_one_found(self):
        tree = MagicMock()
        branch = MagicMock('branch')
        branch.GetName = 'bar'
        tree.GetListOfBranches = [branch]
        print OH.find_branches_matching_pattern(tree, 'bar'), ['bar']
        self.assertEquals(OH.find_branches_matching_pattern(tree, 'bar'), ['bar'])
