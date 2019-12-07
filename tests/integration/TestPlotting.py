import glob
import os
import shutil
import unittest
from PyAnalysisTools.PlottingUtils.Plotter import Plotter
from PyAnalysisTools.base.YAMLHandle import YAMLLoader as yl


class TestBasePlotter(unittest.TestCase):
    def setUp(self):
        self.fixture_path = os.path.join(os.path.dirname(__file__), 'fixtures')
        self.simple_cfg_file = yl.read_yaml(os.path.join(self.fixture_path, 'plotting_test_cfg.yml'))

    def tearDown(self):
        for d in glob.glob('output_*'):
            shutil.rmtree(d)

    def test_simple_plotting(self):
        cfg = self.simple_cfg_file
        plotter = Plotter(input_files=map(lambda fn: os.path.join(self.fixture_path, 'files', fn), cfg['input_files']),
                          process_config_files=os.path.join(self.fixture_path, cfg['merge_cfg']['simple']),
                          xs_config_file=os.path.join(self.fixture_path, cfg['ds_cfg']),
                          plot_config_files=os.path.join(self.fixture_path, cfg['plot_cfg']['simple']),
                          tree_name=cfg['tree_name'],
                          module_config_files=[os.path.join(self.fixture_path, cfg['selection_cfg']['simple'])])
        plotter.make_plots()
        output_path = plotter.output_handle.output_dir
        self.assertTrue(os.path.exists(os.path.join(output_path, 'SR_muon_pt.pdf')))
