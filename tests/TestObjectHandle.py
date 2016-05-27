__author__ = 'marcusmorgenstern'
__mail__ = 'marcus.matthias.morgenstern@cern.ch'

import unittest

import ROOT
from ROOTUtils.FileHandle import FileHandle

import PyAnalysisTools.ROOTUtils.ObjectHandle as OH


class TestObjectHandle(unittest.TestCase):
    def setUp(self):
        pass

    def testGetObjectFromCanvas(self):
        file_handle = FileHandle('test.root')
        file_handle.open()
        canvas = file_handle.get_object_by_name("DRTr-1d_residualDose")
        obj = OH.get_objects_from_canvas(canvas)
        self.assertTrue(len(obj) > 0)

    def testGetObjectFromEmptyCanvas(self):
        ROOT.gROOT.SetBatch(True)
        c = ROOT.TCanvas("c", "c", 800, 600)
        obj = OH.get_objects_from_canvas(c)
        self.assertEqual(len(obj), 0)

    def testObjectFromCanvasByType(self):
        file_handle = FileHandle('test.root')
        file_handle.open()
        canvas = file_handle.get_object_by_name("DRTr-1d_residualDose")
        obj = OH.get_objects_from_canvas_by_type(canvas, "TH2")
        self.assertTrue(len(obj) > 0)

    def testObjectFromCanvasByNonExistingType(self):
        file_handle = FileHandle('test.root')
        file_handle.open()
        canvas = file_handle.get_object_by_name("DRTr-1d_residualDose")
        obj = OH.get_objects_from_canvas_by_type(canvas, "TH1F")
        self.assertEqual(len(obj),  0)
