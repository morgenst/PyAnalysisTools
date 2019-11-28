import unittest
import os
import PyAnalysisTools.AnalysisTools.XSHandle as xs
from PyAnalysisTools.base import InvalidInputError
from mock import MagicMock, Mock

cwd = os.path.dirname(__file__)


class TestXSHandle(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.xs_handle = xs.XSHandle(os.path.join(os.path.dirname(__file__), 'fixtures/dataset_info_pmg.yml'))

    def tearDown(self):
        pass

    def test_uninitialised_cross_section(self):
        self.assertEqual(1., xs.XSHandle(None).get_xs_scale_factor('foo'))

    def test_cross_section_unavailable_process(self):
        self.assertEqual(1., self.xs_handle.get_xs_scale_factor('foo'))

    def test_cross_section_no_kfactor(self):
        self.assertEqual(0.00012676999999999998, self.xs_handle.get_xs_scale_factor('TBsLQee500l1'))

    def test_cross_section_filter_eff(self):
        self.assertEqual(0.9755 * 78420000., self.xs_handle.get_xs_scale_factor('DiJetZ0W'))

    def test_cross_section_xs_info(self):
        self.assertEqual((0.00012676999999999998, 1.0, 1.0), self.xs_handle.retrieve_xs_info('TBsLQee500l1'))

    def test_cross_section_xs_info_unavailable_process(self):
        self.assertEqual((None, 1.0, 1.0), self.xs_handle.retrieve_xs_info('foo'))

    def test_uninitialised_cross_section_invalid(self):
        self.assertRaises(InvalidInputError, xs.XSHandle(None).retrieve_xs_info, 'foo')

    def test_get_lumi_scale_factor(self):
        self.assertEqual(12676.999999999998,
                         self.xs_handle.get_lumi_scale_factor('TBsLQee500l1', 100., 1.))

    def test_get_lumi_scale_factor_fixed_xsec(self):
        self.assertEqual(10000000., self.xs_handle.get_lumi_scale_factor('TBsLQee500l1', 100., 1., 0.1))

    def test_cross_section_get_ds_info(self):
        self.assertEqual('$\\mathrm{LQ}\\rightarrow se (m_{\\mathrmLQ}} = 500~\\GeV)$',
                         self.xs_handle.get_ds_info('TBsLQee500l1', 'latex_label'))

    def test_cross_section_get_ds_info_invalid_process(self):
        self.assertEqual(None, self.xs_handle.get_ds_info('TBsLQee500000l1', 'latex_label'))

    def test_cross_section_get_ds_info_invalid_element(self):
        self.assertEqual(None, self.xs_handle.get_ds_info('TBsLQee500l1', 'label'))

    def test_cross_section_weight(self):
        process = Mock()
        process.process_name = 'TBsLQee500l1'
        self.assertEqual(1267.6999999999998, xs.get_xsec_weight(100., process, self.xs_handle, {process: 10}))

    def test_cross_section_weight_lumi_dict(self):
        process = Mock()
        process.process_name = 'TBsLQee500l1'
        process.mc_campaign = 'mc16a'
        self.assertEqual(1267.6999999999998, xs.get_xsec_weight({'mc16a': 100., 'mc16d': 100.}, process,
                                                                self.xs_handle,
                                                                {process: 10}))

    def test_dataset_defaults(self):
        ds = xs.Dataset()
        self.assertFalse(ds.is_data)
        self.assertFalse(ds.is_mc)

    def test_dataset(self):
        ds = xs.Dataset(name='foo', cross_section='5*2')
        self.assertFalse(ds.is_data)
        self.assertFalse(ds.is_mc)
        self.assertEqual('foo', ds.name)
        self.assertEqual(10, ds.cross_section)

    def test_xs_info(self):
        dataset = MagicMock(spec=[])
        dataset.cross_section = 10.
        dataset.is_data = False
        #dataset.kfactor = MagicMock()
        dataset.name = 'foo'
        xsi = xs.XSInfo(dataset)
        self.assertEqual('foo', xsi.name)
        self.assertEqual(10., xsi.xsec)
        self.assertEqual(1., xsi.kfactor)
        self.assertEqual(1., xsi.filtereff)

    def test_xs_info_data(self):
        dataset = MagicMock()
        dataset.is_data = True
        self.assertEqual(0, len(xs.XSInfo(dataset).__dict__))