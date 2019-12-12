import unittest
from unittest.mock import MagicMock, PropertyMock

from PyAnalysisTools.AnalysisTools.ProcessFilter import ProcessFilter


class TestProcessFilter(unittest.TestCase):
    def setUp(self):
        self.data = {"foo": 1, "bar": 2, "foobar": 3}

    def test_get_object_from_canvas(self):
        filter = ProcessFilter(processes=["foo"])
        data = filter.execute(self.data)
        expected = {"foo": 1, "foobar": 3}
        self.assertDictEqual(data, expected)

    def test_get_object_from_canvas2(self):
        filter = ProcessFilter(processes=["bar"])
        data = filter.execute(self.data)
        expected = {"bar": 2, "foobar": 3}
        self.assertDictEqual(data, expected)

    def test_get_object_from_canvas_empty(self):
        filter = ProcessFilter(processes=[])
        data = filter.execute(self.data)
        self.assertDictEqual(data, {})

    def test_get_object_key_object(self):
        filter = ProcessFilter(processes=['bar'])
        mock_key = MagicMock(name='bar')
        type(mock_key).name = PropertyMock(return_value='bar')
        self.data = {"foo": 1,  mock_key: 2}
        data = filter.execute(self.data)
        expected = {mock_key: 2}
        self.assertDictEqual(data, expected)
