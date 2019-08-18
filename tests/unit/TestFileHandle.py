import unittest

from PyAnalysisTools.ROOTUtils.FileHandle import FileHandle


class TestFileHandle(unittest.TestCase):
    def setUp(self):
        self.handle = FileHandle(file_name='test.root')

    @unittest.skip("Not correctly implemented")
    def testFileOpenNoPathSuccess(self):
        handle = FileHandle(file_name='test.root')
        handle.open()

    @unittest.skip("Not correctly implemented")
    def testFileOpenNoPathFail(self):
        with self.assertRaises(ValueError):
            FileHandle(file_name='NonExistingFile.root').open()

    @unittest.skip("Not correctly implemented")
    def testFileGetObjects(self):
        self.handle.open()
        l = self.handle.get_objects()
        self.assertEqual(len(l), 1)

    @unittest.skip("Not correctly implemented")
    def testFileGetObjectsByTypeCanvas(self):
        self.handle.open()
        l = self.handle.get_objects_by_type("TCanvas")
        self.assertEqual(len(l), 0)

    @unittest.skip("Not correctly implemented")
    def testFileGetObjectByTypeHist(self):
        self.handle.open()
        l = self.handle.get_objects_by_type("TH1F")
        self.assertEqual(len(l), 1)

    @unittest.skip("Not correctly implemented")
    def testFileGetObjectByNameExisting(self):
        self.handle.open()
        obj = self.handle.get_object_by_name("test_hist_1")
        self.assertTrue(obj is not None)

    @unittest.skip("Not correctly implemented")
    def testFileGetObjectByNameNonExisting(self):
        self.handle.open()
        with self.assertRaises(ValueError):
            obj = self.handle.get_object_by_name("SomeNonExistingObj")

