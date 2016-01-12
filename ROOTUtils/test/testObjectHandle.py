__author__ = 'marcusmorgenstern'
__mail__ = 'marcus.matthias.morgenstern@cern.ch'

import unittest
import ROOT

import ROOTUtils.ObjectHandle as OH
from ROOTUtils.FileHandle import FileHandle

class TestObjectHandle(unittest.TestCase):
    def setUp(self):
        pass

    def testGetObjectFromCanvas(self):
        fHandle = FileHandle('test.root')
        fHandle.open()
        canvas = fHandle.getObjectByName("DRTr-1d_residualDose")
        obj = OH.getObjectsFromCanvas(canvas)
        self.assertTrue(len(obj) > 0)

    def testGetObjectFromEmptyCanvas(self):
        ROOT.gROOT.SetBatch(True)
        c = ROOT.TCanvas("c", "c", 800, 600)
        obj = OH.getObjectsFromCanvas(c)
        self.assertEqual(len(obj), 0)

    def testObjectFromCanvasByType(self):
        fHandle = FileHandle('test.root')
        fHandle.open()
        canvas = fHandle.getObjectByName("DRTr-1d_residualDose")
        obj = OH.getObjectsFromCanvasByType(canvas, "TH2")
        self.assertTrue(len(obj) > 0)

    def testObjectFromCanvasByNonExistingType(self):
        fHandle = FileHandle('test.root')
        fHandle.open()
        canvas = fHandle.getObjectByName("DRTr-1d_residualDose")
        obj = OH.getObjectsFromCanvasByType(canvas, "TH1F")
        self.assertEqual(len(obj),  0)
