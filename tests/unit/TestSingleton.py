import unittest
from PyAnalysisTools.base.Singleton import Singleton
from future.utils import with_metaclass


class TestModule(unittest.TestCase):
    def setUp(self):
        pass

    def test_instance(self):
        try:

            class Foo(with_metaclass(Singleton)):
                pass
            foo1 = Foo()
            foo2 = Foo()
            self.assertEqual(foo1, foo2)
        except SyntaxError:
            exit()

    def test_instance_ctor(self):
        class Foo(with_metaclass(Singleton)):
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

        foo1 = Foo(1, foo='foo')
        foo2 = Foo(2, foo='bar')
        self.assertEqual(foo1, foo2)
        self.assertEqual(foo1.kwargs, {'foo': 'foo'})
        self.assertEqual(foo1.args, (1,))
        self.assertEqual(foo2.kwargs, {'foo': 'foo'})
        self.assertEqual(foo2.args, (1,))

