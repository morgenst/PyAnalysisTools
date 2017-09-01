import root_numpy
import numpy as np
import pandas as pd
from PyAnalysisTools.ROOTUtils.FileHandle import FileHandle


class Root2NumpyConverter(object):
    def __init__(self, branches):
        self.branches = branches

    def convert_to_array(self, tree):
        data = root_numpy.tree2array(tree, branches=self.branches,
                                     selection="@object_pt.size()==1")
        return pd.DataFrame(data).values

    def merge(self, signals, bkgs):
        signal = np.concatenate(signals)
        bkg = np.concatenate(bkgs)
        data = np.concatenate((signal, bkg))
        labels = np.append(np.ones(signal.shape[0]), np.zeros(bkg.shape[0]))
        return data, labels


class TrainingReader(object):
    def __init__(self, **kwargs):
        self.input_file = FileHandle(file_name=kwargs["input_file"])
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
