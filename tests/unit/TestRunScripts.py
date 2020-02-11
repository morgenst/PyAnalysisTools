import argparse
import subprocess
import unittest
import os
import sys
import mock
import six

from PyAnalysisTools.PlottingUtils.Plotter import Plotter

from subprocess import call

import pandas
import root_numpy
from multiprocess.pool import Pool

from PyAnalysisTools import base
from PyAnalysisTools.AnalysisTools.CutFlowAnalyser import CutflowAnalyser, ExtendedCutFlowAnalyser
from PyAnalysisTools.AnalysisTools.DatasetPrinter import DatasetPrinter
from PyAnalysisTools.base import InvalidInputError
from PyAnalysisTools.base.FileHandle import FileHandle
from PyAnalysisTools.base.YAMLHandle import YAMLLoader, YAMLDumper
from run_scripts import convert_root2numpy, merge_prw, print_dataset_list, print_hist_contents, run_file_check, \
    setup_analysis_package, run_plotting

cwd = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(cwd, '../../run_scripts'))


def patch(*args, **kwargs):
    pass


class TestExecute(unittest.TestCase):
    @mock.patch('argparse.ArgumentParser.parse_args',
                return_value=argparse.Namespace(input_file_list=['foo'], log_level=None,
                                                selection_config=None))
    @mock.patch.object(CutflowAnalyser, '__init__', patch)
    @mock.patch.object(CutflowAnalyser, '__del__', patch)
    @mock.patch.object(CutflowAnalyser, 'execute', patch)
    @mock.patch.object(CutflowAnalyser, 'print_cutflow_table', patch)
    def test_print_cutflow_cutflowanalyser(self, mock_args):
        from print_cutflow import main
        main(None)

    @mock.patch('argparse.ArgumentParser.parse_args',
                return_value=argparse.Namespace(input_file_list=['foo'], log_level=None,
                                                selection_config=True))
    @mock.patch.object(ExtendedCutFlowAnalyser, '__init__', patch)
    @mock.patch.object(ExtendedCutFlowAnalyser, '__del__', patch)
    @mock.patch.object(ExtendedCutFlowAnalyser, 'execute', patch)
    @mock.patch.object(ExtendedCutFlowAnalyser, 'print_cutflow_table', patch)
    def test_print_cutflow_extended_cutflowanalyser(self, mock_args):
        from print_cutflow import main
        main(None)

    def test_print_cutflow_callable(self):
        self.assertEqual(0, call(['print_cutflow.py', '-h']))

    @unittest.skip('Need to access PMG database')
    @mock.patch('argparse.ArgumentParser.parse_args',
                return_value=argparse.Namespace(input_file='foo', log_level=None,
                                                dataset_decoration=''))
    @mock.patch.object(YAMLLoader, 'read_yaml', patch)
    @mock.patch.object(YAMLDumper, 'dump_yaml', patch)
    def test_convert_pmg_xsec_db(self, mock_args):
        from convert_pmg_xsec_db import main
        main(None)

    def test_convert_pmg_xsec_db_callable(self):
        self.assertEqual(0, call(['convert_pmg_xsec_db.py', '-h']))

    @mock.patch('argparse.ArgumentParser.parse_args',
                return_value=argparse.Namespace(input_files=['foo'], log_level=None, tree_name='foo',
                                                output_path='foo', selection=None, var_list=[], format='json',
                                                mining_fraction=None))
    @mock.patch.object(YAMLLoader, 'read_yaml', patch)
    @mock.patch.object(Pool, 'map', patch)
    @mock.patch.object(FileHandle, '__del__', patch)
    @mock.patch.object(FileHandle, 'open', patch)
    def test_convert_root2numpy(self, mock_args):
        from convert_root2numpy import main
        if six.PY2:
            pass  # needs refactoring of staticmethos
        else:
            main(None)

    @mock.patch.object(root_numpy, 'tree2array', patch)
    @mock.patch.object(pandas.DataFrame, 'to_json', patch)
    def test_convert_root2numpy_convert(self):
        from convert_root2numpy import convert_and_dump
        f = mock.MagicMock()
        f.file_name = 'foo.root'
        convert_and_dump(f, 'foo', '')

    def test_convert_root2numpy_callable(self):
        self.assertEqual(0, call(['convert_root2numpy.py', '-h']))

    @mock.patch('argparse.ArgumentParser.parse_args',
                return_value=argparse.Namespace(log_level=None))
    @mock.patch.object(merge_prw, 'merge', patch)
    def test_merge_prw(self, mock_args):
        merge_prw.main(None)

    # @mock.patch.object(root_numpy, 'tree2array', patch)
    @mock.patch.object(base.Utilities, 'recursive_glob', patch)
    @mock.patch.object(subprocess, 'call', patch)
    @mock.patch.object(os, 'chdir', patch)
    @mock.patch.object(os, 'listdir', lambda x: [])
    def test_merge_prw_merge(self):
        args = mock.MagicMock()
        merge_prw.merge(args)

    def test_merge_prw_merge_missing_input_path(self):
        self.assertRaises(InvalidInputError, merge_prw.merge, None)

    def test_merge_prw_callable(self):
        self.assertEqual(0, call(['merge_prw.py', '-h']))

    # with mock.patch.dict('sys.modules', {'pyAMI': mock.MagicMock()}):
    #     from PyAnalysisTools.AnalysisTools.NTupleAnalyser import NTupleAnalyser
    #     @mock.patch('argparse.ArgumentParser.parse_args',
    #                 return_value=argparse.Namespace(input_path=['foo'], log_level=None, dataset_list='foo',
    #                                                 selection_config=True))
    #     @mock.patch.object(NTupleAnalyser, 'check_valid_proxy', patch)
    #     @mock.patch.object(NTupleAnalyser, '__init__', patch)
    #     @mock.patch.object(NTupleAnalyser, 'run', patch)
    #     @mock.patch.object(YAMLLoader, 'read_yaml', lambda _: {})
    #     def test_analyse_ntuples(self, mock_args):
    #         from run_scripts.analyse_ntuples import main
    #         main(None)

    def test_analyse_ntuples_callable(self):
        with mock.patch.dict('sys.modules', {'pyAMI': mock.MagicMock()}):
            from PyAnalysisTools.AnalysisTools.NTupleAnalyser import NTupleAnalyser
            self.patcher = mock.patch.object(NTupleAnalyser, 'check_valid_proxy')
            self.assertEqual(1, call(['analyse_ntuples.py', '-h']))

    @mock.patch('argparse.ArgumentParser.parse_args',
                return_value=argparse.Namespace(log_level=None))
    @mock.patch.object(DatasetPrinter, '__init__', patch)
    @mock.patch.object(DatasetPrinter, 'pprint', patch)
    def test_print_dataset_list(self, _):
        print_dataset_list.main(None)

    @mock.patch('argparse.ArgumentParser.parse_args',
                return_value=argparse.Namespace(log_level=None, format='foo'))
    @mock.patch.object(DatasetPrinter, '__init__', patch)
    @mock.patch.object(DatasetPrinter, 'pprint', patch)
    def test_print_dataset_list_invalid_choice(self, _):
        print_dataset_list.main(None)

    def test_print_dataset_list_callable(self):
        self.assertEqual(0, call(['print_dataset_list.py', '-h']))

    @mock.patch('argparse.ArgumentParser.parse_args',
                return_value=argparse.Namespace(input_file=None, log_level=None,
                                                directory=None))
    @mock.patch.object(FileHandle, '__init__', patch)
    @mock.patch.object(FileHandle, '__del__', patch)
    @mock.patch.object(FileHandle, 'get_objects_by_type', lambda x, y, z: [])
    def test_print_hist_contents(self, _):
        print_hist_contents.main(None)

    def test_print_hist_contents_callable(self):
        self.assertEqual(0, call(['print_hist_contents.py', '-h']))

    @mock.patch('argparse.ArgumentParser.parse_args',
                return_value=argparse.Namespace(input_file=None, log_level=None,
                                                directory=None))
    @mock.patch.object(FileHandle, '__init__', patch)
    @mock.patch.object(FileHandle, '__del__', patch)
    @mock.patch.object(run_file_check, 'create_test_case', lambda x, y, z: unittest.TestCase)
    def test_run_file_check(self, _):
        run_file_check.main(None)

    def test_run_file_check_callable(self):
        self.assertEqual(0, call(['run_file_check.py', '-h']))

    @mock.patch('argparse.ArgumentParser.parse_args',
                return_value=argparse.Namespace(input_file=None, log_level=None,
                                                name='foo', path='.', short_name='foo'))
    @mock.patch.object(setup_analysis_package.ModuleCreator, 'create', patch)
    def test_setup_analysis_package_check(self, _):
        setup_analysis_package.main(None)

    def test_setup_analysis_package_callable(self):
        self.assertEqual(0, call(['setup_analysis_package.py', '-h']))

    @mock.patch('argparse.ArgumentParser.parse_args',
                return_value=argparse.Namespace(log_level=None, dataset_list=None))
    @mock.patch.object(YAMLLoader, 'read_yaml', lambda _: {})
    def test_get_dataset_size_check(self, *mock_args):
        if six.PY2:
            pass  # needs refactoring of staticmethos
        else:
            with mock.patch.dict('sys.modules', {'pyAMI': mock.MagicMock()}):
                from run_scripts import get_dataset_size
                get_dataset_size.main(None)

    def test_get_dataset_size_callable(self):
        self.assertEqual(1, call(['get_dataset_size.py', '-h']))

    @mock.patch('argparse.ArgumentParser.parse_args',
                return_value=argparse.Namespace(log_level=None, dataset_list=None,
                                                output_dir=None))
    @mock.patch.object(Plotter, '__init__', patch)
    @mock.patch.object(Plotter, 'make_plots', patch)
    def test_run_plotting_check(self, *mock_args):
        run_plotting.main(None)

    def test_run_plotting_callable(self):
        self.assertEqual(0, call(['run_plotting.py', '-h']))
