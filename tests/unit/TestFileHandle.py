import unittest
import os
import ROOT
from PyAnalysisTools.base.FileHandle import FileHandle


class TestFileHandle(unittest.TestCase):
    def setUp(self):
        self.file_name = os.path.join(os.path.dirname(__file__), 'fixtures/files/ntuple-311331_0.MC16d.root')
        self.file_handle = FileHandle(file_name=self.file_name)

    def tearDown(self):
        del self.file_handle

    @unittest.skip("Not correctly implemented")
    def testFileOpenNoPathSuccess(self):
        handle = FileHandle(file_name='test.root')
        handle.open()

    def testFileOpenNoPathFail(self):
        with self.assertRaises(ValueError):
            FileHandle(file_name='NonExistingFile.root').open()

    def testFileGetObjects(self):
        self.file_handle.open()
        objects = self.file_handle.get_objects()
        self.assertGreater(len(objects), 1)

    def testFileGetObjectsByTypeCanvas(self):
        objects = self.file_handle.get_objects_by_type("TCanvas")
        self.assertEqual(len(objects), 0)

    def testFileGetObjectByTypeHist(self):
        objects = self.file_handle.get_objects_by_type("TH1D")
        self.assertGreater(len(objects), 1)

    def testFileGetObjectByNameExisting(self):
        obj = self.file_handle.get_object_by_name("Nominal")
        self.assertIsInstance(obj, ROOT.TDirectoryFile)

    def test_get_object_inexisting_with_tdir(self):
        obj = self.file_handle.get_object_by_name('BaseSelection_lq_tree_syst_Final', "Nominal")
        self.assertIsInstance(obj, ROOT.TTree)

    def test_get_object_inexisting(self):
        with self.assertRaises(ValueError):
            self.file_handle.get_object_by_name("SomeNonExistingObj")

    def test_get_object_inexisting_tdir(self):
        with self.assertRaises(ValueError):
            self.file_handle.get_object_by_name('BaseSelection_lq_tree_syst_Final', "Nominal2")

    def test_branch_names(self):
        objects = self.file_handle.get_branch_names_from_tree('BaseSelection_lq_tree_syst_Final', 'Nominal')
        self.assertGreater(len(objects), 1)

    def test_get_objects_by_pattern(self):
        objects = self.file_handle.get_objects_by_pattern('JET_')
        self.assertGreater(len(objects), 1)

    def test_get_objects_by_pattern_inexisting(self):
        objects = self.file_handle.get_objects_by_pattern('BJET_')
        self.assertEqual(len(objects), 0)

    def test_branch_names_pattern(self):
        objects = self.file_handle.get_branch_names_from_tree('BaseSelection_lq_tree_syst_Final', 'Nominal', 'muon_n')
        self.assertEqual(len(objects), 1)

    def test_branch_names_pattern_inexisting(self):
        objects = self.file_handle.get_branch_names_from_tree('BaseSelection_lq_tree_syst_Final', 'Nominal', 'muon_n2')
        self.assertEqual(len(objects), 0)

    def test_branch_names_pattern_inexisting_tree(self):
        with self.assertRaises(ValueError):
            self.file_handle.get_branch_names_from_tree('tree', 'Nominal', 'muon_n2')

    def test_get_number_of_total_events(self):
        self.assertEqual(self.file_handle.get_number_of_total_events(), 10000.)

    def test_get_number_of_total_events_uneweighted(self):
        self.assertEqual(self.file_handle.get_number_of_total_events(unweighted=True), 10000.)

    def test_get_daod_events(self):
        self.assertEqual(self.file_handle.get_daod_events(), 10000.)

    @unittest.skip
    def test_friend(self):
        pass
        # file_handle = FileHandle(file_name=self.file_name,
        #                          friend_directory=os.path.join(os.path.dirname(__file__), 'fixtures/files/'),
        #                          friend_pattern='ntuple-', friend_tree_names='BaseSelection_lq_tree_syst_Final')

    def test_get_directory(self):
        self.assertIsNotNone(self.file_handle.get_directory('Nominal'))

    @unittest.skip("Segfault in py3")
    def test_get_directory_fail(self):
        self.assertRaises(TypeError, self.file_handle.get_directory('Nominal2'))
