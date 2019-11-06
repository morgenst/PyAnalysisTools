#!/usr/bin/env python
import unittest

from PyAnalysisTools.base import get_default_argparser, default_init
from PyAnalysisTools.base.FileHandle import FileHandle as fh


def create_test_case(_, input_fn, reference_fn):
    class FileChecker(unittest.TestCase):
        """
        Class to compare root file (main purpose is to check output from analysis CI jobs.
        """
        @classmethod
        def setUpClass(self):
            self.input = fh(file_name=input_fn, switch_off_process_name_analysis=True)
            self.reference = fh(file_name=reference_fn, switch_off_process_name_analysis=False)

        def test_number_of_stored_objects(self):
            self.assertGreaterEqual(len(self.input.get_objects()), len(self.reference.get_objects()))

        def test_tree_and_branch_exist(self):
            for tree in self.reference.get_objects_by_type('TTree', 'Nominal'):
                self.assertIsNotNone(self.input.get_object_by_name(tree.GetName(), 'Nominal'))
                test_branches = self.input.get_branch_names_from_tree(tree.GetName(), 'Nominal')
                for branch in self.reference.get_branch_names_from_tree(tree.GetName(), 'Nominal'):
                    if branch in test_branches:
                        continue
                    self.assert_(False, 'Could not find branch {:s} in test output'.format(branch))

        def test_cutflows(self):
            self.assertEqual(self.reference.get_number_of_total_events(), self.input.get_number_of_total_events())
            self.assertEqual(self.reference.get_daod_events(), self.input.get_daod_events())
            cf_ref = self.reference.get_object_by_name('cutflow_BaseSelection_raw', tdirectory='Nominal')
            cf_input = self.reference.get_object_by_name('cutflow_BaseSelection_raw', tdirectory='Nominal')
            self.assertIsNotNone(cf_ref)
            self.assertIsNotNone(cf_input)
            self.assertEqual(cf_ref.GetNbinsX(), cf_input.GetNbinsX())
            for b in range(cf_ref.GetNbinsX()+1):
                self.assertEqual(cf_ref.GetBinContent(b), cf_input.GetBinContent(b))

    test_case = FileChecker
    return test_case


if __name__ == '__main__':
    parser = get_default_argparser("Execution script of FileChecker unittests")
    parser.add_argument('input_file', help='input file to test')
    parser.add_argument('reference_file', help='reference to which input file is compared')

    args = default_init(parser)
    loader = unittest.TestLoader()
    tests = unittest.TestSuite()
    tests.addTests(loader.loadTestsFromTestCase(create_test_case(*list(vars(args).values()))))
    unittest.TextTestRunner().run(tests)
