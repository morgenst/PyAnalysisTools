import root_numpy
import numpy as np
import pandas as pd
import pickle
import ROOT
from PyAnalysisTools.base import _logger, InvalidInputError
from PyAnalysisTools.ROOTUtils.FileHandle import FileHandle
from PyAnalysisTools.PlottingUtils.PlotConfig import find_process_config, parse_and_build_process_config
from PyAnalysisTools.PlottingUtils.PlotConfig import PlotConfig as pc
import PyAnalysisTools.PlottingUtils.PlottingTools as pt
import PyAnalysisTools.PlottingUtils.Formatting as fm
from PyAnalysisTools.base.OutputHandle import OutputFileHandle
from PyAnalysisTools.PlottingUtils import set_batch_mode
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler


class DataScaler(object):
    def __init__(self, algo="default"):
        self.scale_algo = algo
        self.scaler = None
    @staticmethod
    def get_algos():
        return ["default", "standard", "min_max"]

    def apply_scaling(self, X, y, dump=None, scaler=None):
        if scaler is not None:
            with open(scaler, "r") as fn:
                return pickle.load(fn).transform(X), y

        le = LabelEncoder()
        if y is not None:
            y = le.fit_transform(y)
        if self.scale_algo == "min_max":
            return self.apply_min_max_scaler(X, dump), y
        elif self.scale_algo == "standard":
            return self.apply_standard_scaler(X, dump), y
        elif self.scale_algo == "default":
            return self.apply_min_max_scaler(X, dump), y
        else:
            _logger.error("Invalid scaling algorithm requested: {:s}".format(self.scale_algo))
            raise InvalidInputError()

    @staticmethod
    def apply_standard_scaler(X, dump=None):
        scaler = StandardScaler()
        return DataScaler.apply_scaler(scaler, X, dump)

    @staticmethod
    def apply_min_max_scaler(X, dump=None):
        scaler = MinMaxScaler(feature_range=(0, 1))
        return DataScaler.apply_scaler(scaler, X, dump)

    @staticmethod
    def apply_scaler(scaler, X, dump=None):
        scaler.fit(X)
        if dump is not None:
            with open(dump, "w") as fn:
                pickle.dump(scaler, fn)
        return scaler.transform(X)


class Root2NumpyConverter(object):
    def __init__(self, branches):
        self.branches = branches

    def convert_to_array(self, tree, selection="", max_events=None):
        data = root_numpy.tree2array(tree, branches=self.branches,
                                     selection=selection, start=0, stop=max_events)
        return data

    def merge(self, signals, bkgs):
        signal = np.concatenate(signals)
        bkg = np.concatenate(bkgs)
        data = np.concatenate((signal, bkg))
        labels = np.append(np.ones(signal.shape[0]), np.zeros(bkg.shape[0]))
        return data, labels


class TrainingReader(object):
    def __init__(self, **kwargs):
        self.numpy_input = False
        if len(kwargs["input_file"]) > 1 and kwargs["input_file"][0].endswith(".npy"):
            self.numpy_input = True
            return
        self.input_file = FileHandle(file_name=kwargs["input_file"][0])
        self.signal_tree_names = kwargs["signal_tree_names"]
        self.bkg_tree_names = kwargs["bkg_tree_names"]

    def get_trees(self):
        signal_train_tree_names, bkg_train_tree_names, signal_eval_tree_names, bkg_eval_tree_names = self.parse_tree_names()
        signal_train_trees = self.read_tree(signal_train_tree_names)
        signal_eval_trees = self.read_tree(signal_eval_tree_names)
        bkg_train_trees = self.read_tree(bkg_train_tree_names)
        bkg_eval_trees = self.read_tree(bkg_eval_tree_names)
        return signal_train_trees, bkg_train_trees, signal_eval_trees, bkg_eval_trees

    def read_tree(self, tree_names):
        return [self.input_file.get_object_by_name(tn) for tn in tree_names]

    def parse_tree_names(self):
        if any("re." in name for name in self.signal_tree_names):
            self.expand_tree_names(self.signal_tree_names)
        if any("re." in name for name in self.bkg_tree_names):
            self.expand_tree_names(self.bkg_tree_names)
        signal_train_tree_names = ["train_{:s}".format(signal_tree_name) for signal_tree_name in self.signal_tree_names]
        background_train_tree_names = ["train_{:s}".format(background_tree_name) for background_tree_name in
                                           self.bkg_tree_names]
        signal_eval_tree_names = ["eval_{:s}".format(signal_tree_name) for signal_tree_name in
                                       self.signal_tree_names]
        background_eval_tree_names = ["eval_{:s}".format(background_tree_name) for background_tree_name in
                                           self.bkg_tree_names]
        return signal_train_tree_names, background_train_tree_names, signal_eval_tree_names, background_eval_tree_names

    def expand_tree_names(self, tree_names):
        for tree_name in tree_names:
            if not tree_name.startswith("re."):
                continue
            pattern = "train_" + tree_name.replace("re.", "").replace("*", ".*")
            tree_names += list(set(map(lambda name: str.replace(name, "train_", ""),
                                       map(lambda obj: obj.GetName(), self.input_file.get_objects_by_pattern(pattern)))))
            tree_names.remove(tree_name)


class MLAnalyser(object):
    def __init__(self, **kwargs):
        kwargs.setdefault("batch", True)
        self.file_handles = [FileHandle(file_name=file_name, dataset_info=kwargs["cross_section_config"])
                             for file_name in kwargs["input_files"]]
        self.process_config_file = kwargs["process_config_file"]
        self.branch_name = kwargs["branch_name"]
        self.tree_name = kwargs["tree_name"]
        self.converter = Root2NumpyConverter(self.branch_name)
        self.process_configs = parse_and_build_process_config(self.process_config_file)
        self.output_handle = OutputFileHandle(output_dir=kwargs["output_dir"])
        set_batch_mode(kwargs["batch"])

    def parse_process_config(self):
        if self.process_config_file is None:
            return None
        process_config = parse_and_build_process_config(self.process_config_file)
        return process_config

    def read_score(self):
        trees = {fh.process: fh.get_object_by_name(self.tree_name, "Nominal") for fh in self.file_handles}
        arrays = {process: self.converter.convert_to_array(tree) for process, tree in trees.iteritems()}
        signals = []
        backgrounds = []

        for process_name in trees.keys():
            _ = find_process_config(process_name, self.process_configs)
        for process, process_config in self.process_configs.iteritems():
            if not hasattr(process_config, "subprocesses"):
                continue
            for sub_process in process_config.subprocesses:
                if sub_process not in arrays.keys():
                    continue
                if process_config.type.lower() == "signal":
                    signals.append(arrays[sub_process])
                elif process_config.type.lower() == "background" or process_config.type.lower() == "data":
                    backgrounds.append(arrays[sub_process])
                else:
                    print "Could not classify {:s}".format(sub_process)
        signal = np.concatenate(signals)
        background = np.concatenate(backgrounds)
        return signal + 1., background + 1.

    def plot_roc(self):
        signal, background = self.read_score()
        efficiencies = [100. - i * 10. for i in range(10)]
        cuts = [np.percentile(signal, eff) for eff in efficiencies]
        signal_total = sum(signal)
        signal_eff = [np.sum(signal[signal > cut] / signal_total) for cut in cuts]

        bkg_total = sum(background)
        bkg_rej = [1. - np.sum(background[background > cut] / bkg_total) for cut in cuts]
        curve = ROOT.TGraph(len(efficiencies))
        curve.SetName("roc_curve")
        for b in range(len(efficiencies)):
            rej = bkg_rej[b]
            if rej == np.inf:
                rej = 0
            curve.SetPoint(b, signal_eff[b], rej)
        plot_config = pc(name="roc_curve", xtitle="signal efficiency", ytitle="background rejection", draw="Line",
                         logy=True, watermark="Internal", lumi=1.)
        canvas = pt.plot_obj(curve, plot_config)
        fm.decorate_canvas(canvas, plot_config)
        self.output_handle.register_object(canvas)
        self.output_handle.write_and_close()
