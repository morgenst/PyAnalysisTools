from __future__ import division

import os
import pickle

import numpy as np
import pandas as pd
import root_numpy
from builtins import object
from builtins import range
from past.utils import old_div
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report
import PyAnalysisTools.PlottingUtils.Formatting as fm
import PyAnalysisTools.PlottingUtils.PlottingTools as pt
import ROOT
from PyAnalysisTools.PlottingUtils import set_batch_mode
from PyAnalysisTools.PlottingUtils.PlotConfig import PlotConfig as pc
from PyAnalysisTools.PlottingUtils.PlotConfig import find_process_config, parse_and_build_process_config
from PyAnalysisTools.base.FileHandle import FileHandle
from PyAnalysisTools.base import _logger, InvalidInputError
from PyAnalysisTools.base.OutputHandle import OutputFileHandle
from PyAnalysisTools.base.ShellUtils import make_dirs
import matplotlib
matplotlib.use('Agg')  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def explode(df, lst_cols, fill_value=''):
    # make sure `lst_cols` is a list
    if lst_cols and not isinstance(lst_cols, list):
        lst_cols = [lst_cols]
    # all columns except `lst_cols`
    idx_cols = df.columns.difference(lst_cols)

    # calculate lengths of lists
    lens = df[lst_cols[0]].str.len()

    if (lens > 0).all():
        # ALL lists in cells aren't empty
        return pd.DataFrame({col: np.repeat(df[col].values, df[lst_cols[0]].str.len())
                             for col in idx_cols}).assign(
            **{col: np.concatenate(df[col].values) for col in lst_cols}).loc[:, df.columns]
    else:
        # at least one list in cells is empty
        return pd.DataFrame({col: np.repeat(df[col].values, df[lst_cols[0]].str.len())
                             for col in idx_cols}).assign(
            **{col: np.concatenate(df[col].values)
               for col in lst_cols}).append(df.loc[lens == 0, idx_cols]).fillna(fill_value).loc[:, df.columns]


def print_classification(model, X, y, Xeval, yeval, output_path=None):
    class_preds_train = model.predict_classes(X)
    class_preds_eval = model.predict_classes(Xeval)

    report_train = classification_report(class_preds_train, y)
    print('########## on same dataset (bias) ##########')
    print(report_train)

    report_unbiased = classification_report(class_preds_eval, yeval)
    print('\n\n########## on independent dataset (unbiased) ##########')
    print(report_unbiased)

    if output_path is not None:
        store = os.path.join(output_path, 'classification_reports')
        make_dirs(store)
        with open(os.path.join(store, 'training_set.txt'), 'w') as f:
            print(report_train, file=f)
        with open(os.path.join(store, 'test_set.txt'), 'w') as f:
            print(report_unbiased, file=f)


def plot_scoring(history, name, scorers, output_path):
    """
    Make scoring plots for each epoch, i.e. loss, accuracy etc
    :param history: training history
    :param name: output name
    :param scorers: scorings to be plotted
    :param output_path: output path
    :return:
    """
    for scorer in scorers:
        plt.plot(history.history[scorer])
        plt.plot(history.history['val_{:s}'.format(scorer)])
    plt.legend([i for x in [(scorer, "valid {:s}".format(scorer)) for scorer in scorers] for i in x])
    plt.xlabel('epoch')
    plt.ylabel('scoring')
    store = os.path.join(output_path, 'plots')
    make_dirs(store)
    plt.savefig(os.path.join(store, '{:s}.png'.format(name)))
    plt.savefig(os.path.join(store, '{:s}.pdf'.format(name)))
    plt.close()


class MLConfig(object):
    """
    Class containing configration of ML classifier
    """

    def __init__(self, **kwargs):
        kwargs.setdefault('scaler', None)
        self.score_name = kwargs['branch_name']
        self.varset = kwargs['variable_list']
        self.scaler = kwargs['scaler']
        self.selection = kwargs['selection']

    def __str__(self):
        """
        Overloaded str operator. Get's called if object is printed
        :return: formatted string with name and attributes
        :rtype: str
        """
        obj_str = "Attached ML branch {:s} was created with the following configuration \n".format(self.score_name)
        obj_str += 'variables: \n'
        for var in self.varset:
            obj_str += '\t {:s}\n'.format(var)
        if self.selection is not None:
            obj_str += 'selection: \n'
            for sel in self.selection:
                obj_str += '\t {:s}\n'.format(sel)
        else:
            obj_str += 'selection: None\n'
        obj_str += 'scaler: {:s}'.format(self.scaler)
        return obj_str

    def __eq__(self, other):
        """
        Comparison operator
        :param other: ML config object to compare to
        :type other: MLConfig
        :return: True/False
        :rtype: boolean
        """
        if isinstance(self, other.__class__):
            for k, v in list(self.__dict__.items()):
                if k not in other.__dict__:
                    return False
                if k == 'scaler':
                    if self.__dict__[k].scale_algo != other.__dict__[k].scale_algo:
                        return False
                    continue
                if self.__dict__[k] != other.__dict__[k]:
                    return False
            return True
        return False

    def __ne__(self, other):
        """
        Comparison operator (negative)
        :param other: ML config object to compare to
        :type other: MLConfig
        :return: True/False
        :rtype: boolean
        """
        return not self.__eq__(other)


class MLConfigHandle(object):
    """
    Handle to create and add ML configuration to summary file in friend directory
    """

    def __init__(self, **kwargs):
        self.config = MLConfig(**kwargs)
        self.output_path = kwargs['output_path']
        self.file_name = os.path.join(self.output_path, 'ml_config_summary.pkl')

    def dump_config(self):
        data = {}
        if os.path.exists(self.file_name):
            with open(self.file_name, 'r') as f:
                data = pickle.load(f)
        if self.config.score_name in data:
            if self.config == data[self.config.score_name]:
                return
            _logger.error('Score with name {:s} does already exist, but has different config. '
                          'Will give up adding it'.format(self.config.score_name))
            exit()
        data[self.config.score_name] = self.config
        with open(self.file_name, 'w') as f:
            pickle.dump(data, f)


class DataScaler(object):
    def __init__(self, algo="default"):
        self.scale_algo = algo
        self.scaler = None

    @staticmethod
    def get_algos():
        return ["default", "standard", "min_max"]

    def apply_scaling(self, X, y, dump=None, scaler=None):
        if scaler is not None:
            with open(scaler, 'rb') as fn:
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
            _logger.debug('Store scaler to: {:s}'.format(dump))
            with open(dump, "wb") as fn:
                pickle.dump(scaler, fn)
        return scaler.transform(X)


class Root2NumpyConverter(object):
    def __init__(self, branches):
        self.branches = branches

    def convert_to_array(self, tree, selection="", max_events=None):
        selection = "1"
        data = root_numpy.tree2array(tree, branches=self.branches, selection=None, start=0)
        return data

    def merge(self, signals, bkgs):
        signal = np.concatenate(signals)
        bkg = np.concatenate(bkgs)
        data = np.concatenate((signal, bkg))
        labels = np.append(np.ones(signal.shape[0]), np.zeros(bkg.shape[0]))
        return data, labels


class TrainingReader(object):
    def __init__(self, **kwargs):
        def check_file_type(postfix):
            return all([i.endswith(postfix) for i in kwargs['input_file_list']])

        self.mode = ''
        self.numpy_input = False
        if 'input_file' in kwargs:
            if len(kwargs["input_file"]) > 1 and kwargs["input_file"][0].endswith(".npy"):
                self.numpy_input = True
                return
            self.input_file = FileHandle(file_name=kwargs["input_file"][0])
            self.signal_tree_names = kwargs["signal_tree_names"]
            self.bkg_tree_names = kwargs["bkg_tree_names"]

        if 'input_file_list' in kwargs:
            if check_file_type('.json'):
                self.mode = 'pandas'
                self.data = {}
                for fn in kwargs['input_file_list']:
                    with open(fn) as f:
                        self.data[fn.split('/')[-1]] = pd.read_json(f)

    def get_trees(self):
        signal_train_tree_names, bkg_train_tree_names, signal_eval_tree_names, bkg_eval_tree_names = \
            self.parse_tree_names()
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
        expanded_tree_names = []
        tree_names_to_remove = []
        for tree_name in tree_names:
            if not tree_name.startswith("re."):
                continue
            pattern = "train_" + tree_name.replace("re.", "").replace("*", ".*")
            expanded_tree_names += list(set([str.replace(name, "train_", "")
                                             for name in [obj.GetName()
                                                          for obj in self.input_file.get_objects_by_pattern(pattern)]]))
            tree_names_to_remove.append(tree_name)
        for tree_name in tree_names_to_remove:
            tree_names.remove(tree_name)
        tree_names += expanded_tree_names


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

    def read_score(self, selection=None):
        trees = {fh.process: fh.get_object_by_name(self.tree_name, "Nominal") for fh in self.file_handles}
        arrays = {process: self.converter.convert_to_array(tree, selection=selection) for process, tree in
                  list(trees.items())}
        signals = []
        backgrounds = []

        for process_name in list(trees.keys()):
            _ = find_process_config(process_name, self.process_configs)
        for process, process_config in list(self.process_configs.items()):
            if not hasattr(process_config, "subprocesses"):
                continue
            for sub_process in process_config.subprocesses:
                if sub_process not in list(arrays.keys()):
                    continue
                if process_config.type.lower() == "signal":
                    signals.append(arrays[sub_process])
                elif process_config.type.lower() == "background" or process_config.type.lower() == "data":
                    backgrounds.append(arrays[sub_process])
                else:
                    _logger.warn("Could not classify {:s}".format(sub_process))

        signal = np.concatenate(signals)
        background = np.concatenate(backgrounds)
        return signal + 1., background + 1.

    def plot_roc(self, selection=None):
        signal, background = self.read_score(selection)
        efficiencies = [100. - i * 10. for i in range(10)]
        for eff in efficiencies:
            _logger.debug("eff: {:.2f} bkg eff: {:.2f} rej: {:.2f}".format(eff,
                                                                           np.percentile(signal, eff),
                                                                           np.percentile(signal, 100. - eff)))
        cuts = [np.percentile(signal, 100. - eff) for eff in efficiencies]
        signal_total = sum(signal)
        signal_eff = [np.sum(old_div(signal[signal < cut], signal_total)) for cut in cuts]

        bkg_total = sum(background)
        bkg_rej = [1. - np.sum(old_div(background[background < cut], bkg_total)) for cut in cuts]
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
