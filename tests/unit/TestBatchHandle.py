import glob
import os
import time
import unittest
from PyAnalysisTools.base import BatchHandle as bh, ShellUtils
from mock import patch


class TestJob(bh.BatchJob):
    def __init__(self, **kwargs):
        kwargs.setdefault('log_level', 'info')
        kwargs.setdefault('local', False)
        kwargs.setdefault('submit_args', ['foo', 'foo', 1, 'foo'])
        for k, v in kwargs.items():
            setattr(self, k, v)

    def finish(self, *args, **kwargs):
        pass

    def prepare(self, *args, **kwargs):
        pass

    def run(self, *args, **kwargs):
        pass


class Patch(object):
    def __init__(self, *args, **kwargs): None

    def __enter__(self, *args, **kwargs): None

    def __exit__(self, *args, **kwargs): None


class TestBatchHandle(unittest.TestCase):
    def setUp(self):
        self.master_test_job = TestJob(cluster_cfg_file=None, queue='foo')

    def test_batch_job(self):
        self.assertIsNotNone(bh.BatchJob())

    @patch.object(time, 'sleep', lambda _: None)
    def test_ctor(self):
        handle = bh.BatchHandle(self.master_test_job)
        self.assertEqual('qsub', handle.system)
        self.assertEqual('foo', handle.queue)
        self.assertEqual('info', handle.log_level)
        self.assertEqual(self.master_test_job, handle.job)
        self.assertFalse(handle.local)
        self.assertTrue(handle.is_master)

    @patch.object(ShellUtils, 'change_dir', lambda _: True)
    @patch.object(os, 'system', lambda _: None)
    @patch.object(os, 'remove', lambda _: None)
    @patch.object(glob, 'glob', lambda _: ['foo'])
    @patch.object(time, 'sleep', lambda _: None)
    def test_dtor(self):
        handle = bh.BatchHandle(self.master_test_job)
        handle.is_master = True
        handle.output_dir = 'foo'
        self.assertIsNone(handle.__del__())

    @patch.object(time, 'sleep', lambda _: None)
    def test_dtor_child(self):
        handle = bh.BatchHandle(self.master_test_job)
        handle.is_master = False
        self.assertIsNone(handle.__del__())

    @patch.object(os, 'system', lambda _: None)
    @patch.object(time, 'sleep', lambda _: None)
    @patch.object(glob, 'glob', lambda _: ['foo', 'foo'])
    @patch.object(os, 'remove', lambda _: None)
    @patch("builtins.open", new_callable=lambda x=None, y=None: Patch)
    def test_execute(self, _):
        handle = bh.BatchHandle(self.master_test_job)
        os.environ['AnaPySetup'] = 'foo'
        handle.execute()

    @patch.object(os, 'system', lambda _: None)
    @patch.object(time, 'sleep', lambda _: None)
    @patch.object(glob, 'glob', lambda _: ['foo', 'foo'])
    @patch.object(os, 'remove', lambda _: None)
    @patch("builtins.open", new_callable=lambda x=None, y=None: Patch)
    def test_execute_local(self, _):
        self.master_test_job.local = True
        handle = bh.BatchHandle(self.master_test_job)
        os.environ['AnaPySetup'] = 'foo'
        handle.execute()

    @patch.object(os, 'system', lambda _: None)
    @patch.object(time, 'sleep', lambda _: None)
    @patch.object(glob, 'glob', lambda _: ['foo', 'foo'])
    @patch.object(os, 'remove', lambda _: None)
    @patch("builtins.open", new_callable=lambda x=None, y=None: Patch)
    def test_submit_child_skip(self, _):
        os.environ.pop('AnaPySetup')
        handle = bh.BatchHandle(self.master_test_job)
        self.assertRaises(SystemExit, handle.submit_childs, 'foo', 'foo', 2, 'foo', True, [1])
