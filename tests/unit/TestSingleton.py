import six
import unittest
from PyAnalysisTools.base.Singleton import Singleton


class TestModule(unittest.TestCase):
    def setUp(self):
        pass

    @unittest.skipIf(six.PY2, 'Python3 test running python2')
    def test_instance(self):
        class Foo(metaclass=Singleton):
            pass
        foo1 = Foo()
        foo2 = Foo()
        self.assertEqual(foo1, foo2)

    @unittest.skipIf(six.PY2, 'Python3 test running python2')
    def test_instance_ctor(self):
        class Foo(metaclass=Singleton):
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

    @unittest.skipIf(six.PY3, 'Python2 test running python3')
    def test_instance_py2(self):
        class Foo(object):
            __metaclass__ = Singleton
            pass

        foo1 = Foo()
        foo2 = Foo()
        self.assertEqual(foo1, foo2)

    @unittest.skipIf(six.PY3, 'Python2 test running python3')
    def test_instance_ctor_py2(self):
        class Foo(object):
            __metaclass__ = Singleton
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

