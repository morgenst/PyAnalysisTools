import os
import unittest
from PyAnalysisTools.base import Utilities, InvalidInputError
#from pyfakefs.fake_filesystem_unittest import TestCase

cwd = os.path.dirname(__file__)


class TestUtilities(unittest.TestCase):
    def setUp(self):
        pass
        #self.setUpPyfakefs()

    def tearDown(self):
        pass
        #self.tearDownPyfakefs()

    def test_merge_dicts(self):
        d1 = {'foo': 1}
        d2 = {'bar': 2}
        self.assertEqual({'foo': 1, 'bar': 2}, Utilities.merge_dictionaries(d1, d2))

    def test_merge_dicts_single(self):
        d1 = {'foo': 1}
        self.assertEqual(d1, Utilities.merge_dictionaries(d1))

    def test_merge_dicts_fail(self):
        d1 = {'foo': 1}
        d2 = ['bar', 2]
        self.assertEqual({'foo': 1}, Utilities.merge_dictionaries(d1, d2))

    def test_check_required_args_found(self):
        self.assertIsNone(Utilities.check_required_args('arg', arg=1))

    def test_check_required_args_missing(self):
        self.assertEqual('arg', Utilities.check_required_args('arg', foo=1))

    @unittest.skip("Requires fake fs")
    def test_cleaner_check_lifetime(self):
        self.fs.create_file('/foo/bar.txt')
        self.assertTrue(Utilities.Cleaner.check_lifetime(100, 'foo', ['bar.txt']))

    def test_flatten_single_element(self):
        self.assertEqual(['foo/bar/1'], Utilities.flatten({'foo': {'bar': ["1"]}}))

    def test_flatten_more_elements(self):
        self.assertEqual(['foo/bar/1', 'foo/bar/2'], Utilities.flatten({'foo': {'bar': ["1", "2"]}}))

    @unittest.skip("Requires fake fs")
    def test_cleaner_default_ctor(self):
        cleaner = Utilities.Cleaner(base_path='foo')
        self.assertTrue(cleaner.safe)
        self.assertEqual('/foo', cleaner.base_path)
        self.assertEqual([".git", ".keep", ".svn", "InstallArea", "RootCoreBin", "WorkArea"], cleaner.keep_pattern)
        self.assertEqual([], cleaner.deletion_list)
        self.assertEqual(14., cleaner.touch_threshold_days)
        self.assertEqual(None, cleaner.trash_path)

    @unittest.skip("Requires fake fs")
    def test_cleaner_default_ctor_trash(self):
        cleaner = Utilities.Cleaner(base_path='foo', trash_path='bar')
        self.assertEqual('bar', cleaner.trash_path)

    def test_cleaner_default_ctor_missing_arg(self):
        self.assertRaises(InvalidInputError, Utilities.Cleaner)

    def test_cleaner_default_setup_trash(self):
        cleaner = Utilities.Cleaner(base_path='foo', safe=False)
        self.assertIsNone(cleaner.setup_temporary_trash())
