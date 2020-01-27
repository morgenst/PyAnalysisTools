import os
import unittest
from builtins import range

import matplotlib
import mock
import numpy as np
import pandas as pd
import root_numpy
from mock import MagicMock, patch, mock_open
import six

import ROOT
from PyAnalysisTools.AnalysisTools import MLHelper as mh
from PyAnalysisTools.base import InvalidInputError
from PyAnalysisTools.base.FileHandle import FileHandle

if six.PY2:
    builtin = '__builtin__'
else:
    builtin = 'builtins'

cwd = os.path.dirname(__file__)
ROOT.gROOT.SetBatch(True)


class TestMLHelpers(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_print_classification(self):
        model = MagicMock()
        model.predict_classes = MagicMock(return_value=[1])
        mh.print_classification(model, 1, [2], 3, [4])

    @mock.patch.object(matplotlib.pyplot, 'savefig', lambda x: None)
    def test_plot_scoring(self):
        class Object(object):
            pass
        history = Object()
        history.history = {'foo': [(100, 100), (200, 200)]}
        history.history['val_foo'] = history.history['foo']
        mh.plot_scoring(history, 'foo', ['foo'], 'foo')


class TestMLConfig(unittest.TestCase):
    def test_ctor_default(self):
        self.assertRaises(KeyError, mh.MLConfig)

    def test_ctor(self):
        cfg = mh.MLConfig(branch_name='foo', variable_list=[], selection=None)
        self.assertEqual('foo', cfg.score_name)
        self.assertEqual([], cfg.varset)
        self.assertIsNone(cfg.scaler)
        self.assertIsNone(cfg.scale_algo)
        self.assertIsNone(cfg.selection)

    def test_equality(self):
        cfg = mh.MLConfig(branch_name='foo', variable_list=[], selection=None)
        cfg2 = mh.MLConfig(branch_name='foo', variable_list=[], selection=None)
        self.assertEqual(cfg, cfg2)

    def test_inequality(self):
        cfg = mh.MLConfig(branch_name='foo', variable_list=[], selection=None)
        cfg2 = mh.MLConfig(branch_name='bar', variable_list=[], selection=None)
        self.assertNotEqual(cfg, cfg2)

    def test_inequality_scaler(self):
        scaler = mh.DataScaler()
        cfg = mh.MLConfig(branch_name='foo', variable_list=[], selection=None, scaler=scaler)
        cfg2 = mh.MLConfig(branch_name='bar', variable_list=[], selection=None)
        self.assertNotEqual(cfg, cfg2)
        self.assertNotEqual(cfg2, cfg)

    def test_inequality_scaler_algo(self):
        scaler_def = mh.DataScaler()
        scaler = mh.DataScaler('foo')
        cfg = mh.MLConfig(branch_name='foo', variable_list=[], selection=None, scaler=scaler)
        cfg2 = mh.MLConfig(branch_name='bar', variable_list=[], selection=None, scaler=scaler_def)
        self.assertNotEqual(cfg, cfg2)
        self.assertNotEqual(cfg2, cfg)

    def test_inequality_type(self):
        cfg = mh.MLConfig(branch_name='foo', variable_list=[], selection=None)
        self.assertNotEqual(cfg, 5.)

    def test_handle_ctor(self):
        handle = mh.MLConfigHandle(branch_name='foo', variable_list=[], selection=None)
        cfg = mh.MLConfig(branch_name='foo', variable_list=[], selection=None)
        self.assertEqual(cfg, handle.config)
        self.assertEqual('.', handle.output_path)
        self.assertEqual('./ml_config_summary.pkl', handle.file_name)

    def test_print(self):
        handle = mh.MLConfig(branch_name='foo', variable_list=['foo'], selection=['bar'])
        print_out = 'Attached ML branch foo was created with the following configuration \nvariables: \n\t foo\n' \
                    'selection: \n\t bar\nscaler: None\n'
        self.assertEqual(print_out, handle.__str__())


class TestRootNumpyConverter(unittest.TestCase):
    def test_ctor(self):
        converter = mh.Root2NumpyConverter(['foo'])
        self.assertEqual(['foo'], converter.branches)

    def test_ctor_no_list(self):
        converter = mh.Root2NumpyConverter('foo')
        self.assertEqual(['foo'], converter.branches)

    def test_merge(self):
        arr1 = np.array([1, 2])
        arr2 = np.array([3, 4])
        arr3 = np.array([5, 6])
        arr4 = np.array([7, 8])
        converter = mh.Root2NumpyConverter('foo')
        data, labels = converter.merge([arr1, arr2], [arr3, arr4])
        np.testing.assert_array_equal(np.array([i+1 for i in range(8)]), data)
        np.testing.assert_array_equal(np.array([1]*4+[0]*4), labels)

    @patch.object(root_numpy, 'tree2array', lambda x, **kwargs: (x, kwargs))  # x, branches, selection: [x, branches, selection])
    def test_convert(self):
        converter = mh.Root2NumpyConverter(['foo'])
        data = converter.convert_to_array(None, 'sel', 1000)
        self.assertIsNone(data[0])
        self.assertEqual({'branches': ['foo'], 'selection': 'sel', 'start': 0, 'stop': 1000}, data[1])


class TestTrainingReader(unittest.TestCase):
    def test_default_ctor(self):
        converter = mh.TrainingReader()
        self.assertEqual('', converter.mode)
        self.assertFalse(converter.numpy_input)

    @mock.patch.object(pd, 'read_json', lambda _: None)
    @patch(builtin + ".open", new_callable=mock_open)
    def test_ctor_json(self, _):
        converter = mh.TrainingReader(input_file_list=['foo.json'])
        self.assertEqual('pandas', converter.mode)
        self.assertFalse(converter.numpy_input)
        self.assertEqual({'foo.json': None}, converter.data)

    def test_ctor_numpy_list(self):
        converter = mh.TrainingReader(input_file=['foo.npy', 'bar.npy'])
        self.assertEqual('', converter.mode)
        self.assertTrue(converter.numpy_input)

    def test_ctor_numpy(self):
        converter = mh.TrainingReader(input_file='foo.npy', signal_tree_names=['sig'], bkg_tree_names=['bkg'])
        self.assertEqual('', converter.mode)
        self.assertFalse(converter.numpy_input)
        self.assertEqual(['sig'], converter.signal_tree_names)
        self.assertEqual(['bkg'], converter.bkg_tree_names)

    def test_parse_tree_names(self):
        converter = mh.TrainingReader(input_file='foo.npy', signal_tree_names=['sig'], bkg_tree_names=['bkg'])
        sig_train, bkg_train, sig_eval, bkg_eval = converter.parse_tree_names()
        self.assertEqual(['train_sig'], sig_train)
        self.assertEqual(['eval_sig'], sig_eval)
        self.assertEqual(['train_bkg'], bkg_train)
        self.assertEqual(['eval_bkg'], bkg_eval)

    @mock.patch.object(FileHandle, 'get_object_by_name', lambda _, x: x)
    def test_get_trees(self):
        converter = mh.TrainingReader(input_file='foo.npy', signal_tree_names=['sig'], bkg_tree_names=['bkg'])
        sig_train, bkg_train, sig_eval, bkg_eval = converter.get_trees()
        self.assertEqual(['train_sig'], sig_train)
        self.assertEqual(['eval_sig'], sig_eval)
        self.assertEqual(['train_bkg'], bkg_train)
        self.assertEqual(['eval_bkg'], bkg_eval)


class TestMLAnalyser(unittest.TestCase):
    def test_default_ctor(self):
        analyser = mh.MLAnalyser(input_files=[])
        self.assertEqual([], analyser.file_handles)
        self.assertIsNone(analyser.process_config_file)
        self.assertIsNone(analyser.process_configs)
        self.assertIsNone(analyser.tree_name)
        self.assertIsNone(analyser.branch_name)

    @mock.patch.object(mh.MLAnalyser, 'read_score', lambda x, y: [np.array([1, 2, 3]), np.array([4, 5, 6])])
    def test_plot_roc(self):
        analyser = mh.MLAnalyser(input_files=[])
        analyser.plot_roc()

    # def test_read_score(self):
    #     analyser = mh.MLAnalyser(input_files=[])
    #     analyser.process_configs
    #     sig, bkg = analyser.read_score()


class TestDataScaler(unittest.TestCase):
    def test_default_ctor(self):
        scaler = mh.DataScaler()
        self.assertEqual('default', scaler.scale_algo)
        self.assertIsNone(scaler.scaler)

    def test_equality(self):
        scaler = mh.DataScaler()
        scaler2 = mh.DataScaler()
        self.assertEqual(scaler, scaler2)
        self.assertEqual(scaler, scaler)
        self.assertEqual(scaler2, scaler)

    def test_inequality(self):
        scaler = mh.DataScaler()
        scaler2 = mh.DataScaler(algo='foo')
        self.assertNotEqual(scaler, scaler2)
        self.assertNotEqual(scaler2, scaler)
        self.assertNotEqual(scaler, 5)

    def test_get_algos(self):
        self.assertEqual(["default", "standard", "min_max"], mh.DataScaler.get_algos())

    def test_apply_scaling_default(self):
        scaler = mh.DataScaler()
        res = scaler.apply_scaling(np.array([1, 2, 3]).reshape(-1, 1), np.array([1, 2, 3]).reshape(-1, 1))
        expected = (np.array([0., 0.5, 1.]).reshape(3, 1), np.array([0, 1, 2]))
        self.assertIsNone(np.testing.assert_array_equal(expected[0], res[0]))
        self.assertIsNone(np.testing.assert_array_equal(expected[1], res[1]))

    def test_apply_scaling_min_max(self):
        scaler = mh.DataScaler(algo='min_max')
        res = scaler.apply_scaling(np.array([1, 2, 3]).reshape(-1, 1), np.array([1, 2, 3]).reshape(-1, 1))
        expected = (np.array([0., 0.5, 1.]).reshape(3, 1), np.array([0, 1, 2]))
        self.assertIsNone(np.testing.assert_array_equal(expected[0], res[0]))
        self.assertIsNone(np.testing.assert_array_equal(expected[1], res[1]))

    def test_apply_scaling_standard(self):
        scaler = mh.DataScaler(algo='standard')
        X = np.array([1, 2, 3]).reshape(-1, 1)
        res = scaler.apply_scaling(X, np.array([1, 2, 3]).reshape(-1, 1))
        expected = ((X - np.mean(X)) / np.std(X), np.array([0, 1, 2]))
        self.assertIsNone(np.testing.assert_array_equal(expected[0], res[0]))
        self.assertIsNone(np.testing.assert_array_equal(expected[1], res[1]))

    def test_apply_scaling_exception(self):
        scaler = mh.DataScaler(algo='foo')
        self.assertRaises(InvalidInputError, scaler.apply_scaling, X=1, y=None)
