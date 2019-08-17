import unittest
from PyAnalysisTools.base import ShellUtils as SU
import os
import shutil


class TestShellUtils(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.path.join(os.path.abspath(os.getcwd()), "test_dir")
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_make_dir(self):
        path = os.path.join(self.test_dir, "foo/bar")
        self.assertFalse(os.path.exists(path))
        SU.make_dirs(path)
        self.assertTrue(os.path.exists(path))

    def test_make_dir_existing(self):
        path = os.path.join(self.test_dir, "foo/bar")
        SU.make_dirs(path)
        SU.make_dirs(path)

    def test_make_dir_existing_no_permission(self):
        self.assertRaises(OSError, SU.make_dirs, "/usr/bin/test_dir")

    def test_move_file(self):
        file_name = "test_file_move.txt"
        f = open(file_name, "w+")
        f.close()
        self.assertFalse(os.path.exists(os.path.join(self.test_dir, file_name)))
        SU.move(file_name, self.test_dir)
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, file_name)))

    def test_move_file_fail_non_existing_file(self):
        self.assertRaises(IOError, SU.move, "non_exiting_file.txt", self.test_dir)

    def test_move_file_fail_non_existing_dir(self):
        file_name = "test_file_move_fail.txt"
        f = open(file_name, "w+")
        f.close()
        self.assertRaises(IOError, SU.move, file_name, os.path.join(self.test_dir, "non_existing_dir/foo"))

    def test_copy_file(self):
        file_name = "test_file_copy.txt"
        f = open(file_name, "w+")
        f.close()
        self.assertFalse(os.path.exists(os.path.join(self.test_dir, file_name)))
        SU.copy(file_name, self.test_dir)
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, file_name)))
        self.assertTrue(os.path.exists(file_name))

    def test_copy_file_fail_non_existing_file(self):
        self.assertRaises(IOError, SU.copy, "non_exiting_file.txt", "dest.txt")

    def test_move_file_fail_non_existing_dir(self):
        file_name = "test_file_copy_fail.txt"
        f = open(file_name, "w+")
        f.close()
        self.assertRaises(IOError, SU.copy, file_name, os.path.join(self.test_dir, "non_existing_dir/foo"))

    def test_source(self):
        file_name = "test.sh"
        f = open(file_name, "w+")
        print >> f, "#!/bin/bash"
        print >> f, "export FOO=test"
        f.close()
        self.assertFalse("FOO" in os.environ)
        SU.source(file_name)
        self.assertTrue("FOO" in os.environ)
        self.assertEqual(os.environ["FOO"], "test")

        # def remove_directory(path, safe=False):
        #     if safe:
        #         try:
        #             os.removedirs(path)
        #         except OSError:
        #             raise
        #     else:
        #         shutil.rmtree(path)

    def test_remove_dir_fail_non_existing(self):
        self.assertRaises(OSError, SU.remove_directory, "non_existing_dir")

    def test_remove_dir(self):
        dir_name = os.path.join(self.test_dir, "test_to_remove")
        os.makedirs(dir_name)
        self.assertTrue(os.path.exists(dir_name))
        SU.remove_directory(dir_name)
        self.assertFalse(os.path.exists(dir_name))

    def test_remove_dir_safe(self):
        dir_name = os.path.join(self.test_dir, "test_to_remove")
        os.makedirs(dir_name)
        self.assertTrue(os.path.exists(dir_name))
        SU.remove_directory(dir_name, True)
        self.assertFalse(os.path.exists(dir_name))

    def test_remove_dir_safe_non_empty(self):
        dir_name = os.path.join(self.test_dir, "test_to_remove")
        os.makedirs(dir_name)
        f = open(os.path.join(dir_name, "file_not_to_delete.txt"), "w+")
        f.close()
        self.assertRaises(OSError, SU.remove_directory, dir_name, True)

    def test_resolve_path_no_symlink(self):
        self.assertEqual(SU.resolve_path_from_symbolic_links(None, "some_path"), "some_path")

    def test_resolve_path_no_relative_path(self):
        self.assertEqual(SU.resolve_path_from_symbolic_links("some_path", None), None)

    def test_resolve_path_abspath(self):
        self.assertEqual(SU.resolve_path_from_symbolic_links("some_path", self.test_dir), self.test_dir)

    def test_resolve_path(self):
        link = os.path.join(self.test_dir, "test_link")
        os.symlink(self.test_dir, link)
        self.assertEqual(SU.resolve_path_from_symbolic_links(link, "../../../tests"), os.getcwd())

