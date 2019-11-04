from builtins import map
from builtins import range
import unittest
import ROOT
import PyAnalysisTools.ROOTUtils.ObjectHandle as oh
from mock import MagicMock

ROOT.gROOT.SetBatch(True)


class TestObjectHandle(unittest.TestCase):
    def setUp(self):
        self.filled_canvas = ROOT.TCanvas('c', 'c')
        self.h = ROOT.TH1F('h', 'h', 100, -1., 1.)
        self.h.FillRandom('gaus', 100)
        self.filled_canvas.cd()
        self.h.Draw()

    def tearDown(self):
        list([c.Close() for c in ROOT.gROOT.GetListOfCanvases()])
        del self.h

    def testGetObjectFromCanvas(self):
        obj = oh.get_objects_from_canvas(self.filled_canvas)
        self.assertTrue(len(obj) > 0)
        self.assertEqual(obj[0], self.h)

    def testGetObjectFromEmptyCanvas(self):
        ROOT.gROOT.SetBatch(True)
        c = ROOT.TCanvas("c", "c", 800, 600)
        obj = oh.get_objects_from_canvas(c)
        self.assertEqual(len(obj), 0)

    def test_get_object_from_canvas_by_ype_th1_success(self):
        obj = oh.get_objects_from_canvas_by_type(self.filled_canvas, "TH1")
        self.assertTrue(len(obj) > 0)
        self.assertEqual(obj[0], self.h)

    def test_get_object_from_canvas_by_ype_th1_wrong_type(self):
        obj = oh.get_objects_from_canvas_by_type(self.filled_canvas, "TH2")
        self.assertTrue(len(obj) == 0)

    def test_get_object_from_canvas_by_ype_th1_list_success(self):
        obj = oh.get_objects_from_canvas_by_type(self.filled_canvas, ["TH1"])
        self.assertTrue(len(obj) > 0)
        self.assertEqual(obj[0], self.h)

    def test_get_object_from_canvas_by_ype_th1_list_wrong_type(self):
        obj = oh.get_objects_from_canvas_by_type(self.filled_canvas, ["TH2"])
        self.assertTrue(len(obj) == 0)

    def test_get_object_from_canvas_by_ype_th1_success(self):
        obj = oh.get_objects_from_canvas_by_name(self.filled_canvas, "h")
        self.assertTrue(len(obj) > 0)
        self.assertEqual(obj[0], self.h)

    def test_get_object_from_canvas_by_ype_th1_wrong_name(self):
        self.assertIsNone(oh.get_objects_from_canvas_by_name(self.filled_canvas, "h2"))

    def test_get_object_from_canvas_by_ype_th1_list_success(self):
        obj = oh.get_objects_from_canvas_by_name(self.filled_canvas, ["h"])
        self.assertTrue(len(obj) > 0)
        self.assertEqual(obj[0], self.h)

    def test_get_object_from_canvas_by_ype_th1_list_wrong_name(self):
        self.assertIsNone(oh.get_objects_from_canvas_by_name(self.filled_canvas, ["h2"]))

    def test_find_branches_matching_pattern_none_found(self):
        tree = MagicMock()
        tree.get_list_of_branches = ['foo']
        self.assertEquals(oh.find_branches_matching_pattern(tree, 'bar'), [])

    def test_find_branches_matching_pattern_one_found(self):
        tree = MagicMock()
        branch = MagicMock('branch')
        branch.GetName = MagicMock(return_value='bar')
        tree.GetListOfBranches = MagicMock(return_value=[branch])
        self.assertEquals(oh.find_branches_matching_pattern(tree, 'bar'), ['bar'])

    def test_merge_objects_by_process_type_no_match(self):
        process_config = [MagicMock()]
        merged_hist = oh.merge_objects_by_process_type(self.filled_canvas, process_config, 'foo')
        for b in range(self.h.GetNbinsX()+1):
            self.assertEqual(self.h.GetBinContent(b), merged_hist.GetBinContent(b))

    def test_merge_objects_by_process_type_no_input(self):
        self.assertIsNone(oh.merge_objects_by_process_type(ROOT.TCanvas(), None, 'foo'))

    def test_merge_objects_by_process_type(self):
        process = MagicMock(return_value='h')
        process.type = 'foo'
        process_config = {'h': process}
        self.filled_canvas.cd()
        self.h.Clone('h_h').Draw('sames')
        self.h.Clone('unc').Draw('sames')
        self.h.Clone('unc').Draw('sames')
        merged_hist = oh.merge_objects_by_process_type(self.filled_canvas, process_config, 'foo')
        for b in range(self.h.GetNbinsX()+1):
            self.assertEqual(2. * self.h.GetBinContent(b), merged_hist.GetBinContent(b))

    def test_merge_objects_by_process_diff_type(self):
        process = MagicMock(return_value='h')
        process.type = 'bar'
        process_config = {'h': process}
        self.filled_canvas.cd()
        self.h.Clone('h_h').Draw('sames')
        merged_hist = oh.merge_objects_by_process_type(self.filled_canvas, process_config, 'foo')
        for b in range(self.h.GetNbinsX()+1):
            self.assertEqual(self.h.GetBinContent(b), merged_hist.GetBinContent(b))