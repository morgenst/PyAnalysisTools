import unittest

from PyAnalysisTools.AnalysisTools.ProcessFilter import ProcessFilter


class TestObjectHandle(unittest.TestCase):
    def setUp(self):
        self.data = {"foo": 1, "bar": 2, "foobar": 3}

    def testGetObjectFromCanvas(self):
        filter = ProcessFilter(processes=["foo"])
        data = filter.execute(self.data)
        expected = {"foo": 1, "foobar": 3}
        self.assertDictEqual(data, expected)

    def testGetObjectFromCanvas2(self):
        filter = ProcessFilter(processes=["bar"])
        data = filter.execute(self.data)
        expected = {"bar": 2, "foobar": 3}
        self.assertDictEqual(data, expected)

    def testGetObjectFromCanvasEmpty(self):
        filter = ProcessFilter(processes=[])
        data = filter.execute(self.data)
        self.assertDictEqual(data, {})
