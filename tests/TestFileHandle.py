__author__ = 'marcusmorgenstern'
__mail__ = ''

import unittest

from PyAnalysisTools.ROOTUtils.FileHandle import FileHandle


class TestFileHandle(unittest.TestCase):
    def setUp(self):
        self.handle = FileHandle('test.root')

    def testFileOpenNoPathSuccess(self):
        handle = FileHandle('test.root')
        handle.open()

    def testFileOpenNoPathFail(self):
        with self.assertRaises(ValueError):
            FileHandle('NonExistingFile.root').open()

    def testFileGetObjects(self):
        self.handle.open()
        l = self.handle.get_objects()
        self.assertEqual(len(l), 9)

    def testFileGetObjectsByTypeCanvas(self):
        self.handle.open()
        l = self.handle.get_objects_by_type("TCanvas")
        self.assertEqual(len(l), 9)

    def testFileGetObjectByTypeHist(self):
        self.handle.open()
        l = self.handle.get_objects_by_type("TH1F")
        self.assertEqual(len(l), 0)

    def testFileGetObjectByNameExisting(self):
        self.handle.open()
        obj = self.handle.get_object_by_name("DRTr-1d_residualDose")
        self.assertTrue(obj is not None)

    def testFileGetObjectByNameNonExisting(self):
        self.handle.open()
        with self.assertRaises(ValueError):
            obj = self.handle.get_object_by_name("DRTr-1s_residualDose")
