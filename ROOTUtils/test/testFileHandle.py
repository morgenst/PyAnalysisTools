__author__ = 'marcusmorgenstern'
__mail__ = ''

import unittest
from ROOTUtils.FileHandle import FileHandle

class TestFileHandle(unittest.TestCase):
    def setUp(self):
        pass

    def testFileOpenNoPathSuccess(self):
        handle = FileHandle('')

    def testFileOpenNoPathFail(self):
        with self.assertRaises(ValueError):
            FileHandle('NonExistingFile.root').open()
