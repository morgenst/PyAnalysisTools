import unittest
import os

from mock import MagicMock

import ROOT
from PyAnalysisTools.base.FileHandle import FileHandle, filter_empty_trees


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

    def test_file_open_None(self):
        self.assertIsNone(FileHandle(file_name=None).open())

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

    def test_exists_true(self):
        self.assertTrue(self.file_handle.exists())

    def test_exists_false(self):
        self.assertFalse(FileHandle(file_name='foobar').exists())

    def test_exists_none(self):
        self.assertFalse(FileHandle(file_name=None).exists())

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

    def test_filter_empty_trees_remove(self):
        tree_mock = MagicMock()
        tree_mock.GetEntries.return_value = 0
        fh_mock = MagicMock()
        fh_mock.get_object_by_name.return_value = tree_mock
        file_handles = [fh_mock]
        file_handles = filter_empty_trees(file_handles, 'foo', None, 'Nominal')
        self.assertEqual([], file_handles)

    def test_filter_empty_trees_keep(self):
        tree_mock = MagicMock()
        tree_mock.GetEntries.return_value = 10
        fh_mock = MagicMock()
        fh_mock.get_object_by_name.return_value = tree_mock
        file_handles = [fh_mock]
        file_handles = filter_empty_trees(file_handles, 'foo', None, 'Nominal')
        self.assertEqual(1, len(file_handles))
