from __future__ import print_function
from __future__ import unicode_literals
from builtins import bytes
from builtins import object
import os
import re
import time
from ROOT import TFile
from builtins import str
from PyAnalysisTools.base import _logger
from PyAnalysisTools.base.ProcessConfig import Process
from PyAnalysisTools.base.ShellUtils import resolve_path_from_symbolic_links, make_dirs, move
from PyAnalysisTools.AnalysisTools.DataStore import DataSetStore
from PyAnalysisTools.PlottingUtils.PlottingTools import project_hist


class FileHandle(object):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("path", "./")
        kwargs.setdefault("cwd", "None")
        kwargs.setdefault("open_option", "READ")
        kwargs.setdefault("run_dir", None)
        kwargs.setdefault("switch_off_process_name_analysis", False)
        kwargs.setdefault('friend_directory', None)
        kwargs.setdefault("friend_pattern", None)
        kwargs.setdefault("friend_tree_names", None)
        kwargs.setdefault("split_mc", False)
        kwargs.setdefault('dataset_info', None)
        if kwargs['file_name'] is not None and not kwargs['file_name'].startswith('root://'):
            self.file_name = resolve_path_from_symbolic_links(kwargs['cwd'], kwargs['file_name'])
            self.path = resolve_path_from_symbolic_links(kwargs['cwd'], kwargs['path'])
            self.absFName = os.path.join(self.path, self.file_name)
        else:
            self.file_name = kwargs['file_name']
            self.path = None
            self.absFName = self.file_name
        self.dataset_info = None
        if 'dataset_info' in kwargs and kwargs['dataset_info'] is not None:
            if isinstance(kwargs['dataset_info'], dict):
                self.dataset_info = kwargs['dataset_info']
            else:
                self.dataset_info = DataSetStore(kwargs['dataset_info']).dataset_info
        self.open_option = kwargs['open_option']
        self.tfile = None
        self.initial_file_name = None
        self.run_dir = kwargs['run_dir']
        self.year = None
        self.period = None
        self.is_data = False
        self.is_cosmics = False
        self.is_mc = False
        self.mc16a = False
        self.mc16c = False
        self.mc16d = False
        self.mc16e = False
        self.mc_campaign = None
        self.friends = None
        self.friend_tree_names = kwargs['friend_tree_names']
        self.friend_pattern = kwargs['friend_pattern']
        self.friend_files = []
        self.switch_off_process_name_analysis = kwargs['switch_off_process_name_analysis']
        if self.dataset_info is None:
            _logger.debug('Turning off process name analysis because no dataset info provided')
            self.switch_off_process_name_analysis = True
        if self.friend_tree_names is not None and not isinstance(self.friend_tree_names, list):
            self.friend_tree_names = [self.friend_tree_names]
        if self.friend_pattern is not None and not isinstance(self.friend_pattern, list):
            self.friend_pattern = [self.friend_pattern]
        if 'ignore_process_name' not in kwargs:
            self.process = Process(self.file_name, self.dataset_info)
            if self.process is not None:
                if self.mc16a:
                    self.process += '.mc16a'
                if self.mc16c:
                    self.process += '.mc16c'
                if self.mc16d:
                    self.process += '.mc16d'
                if self.mc16e:
                    self.process += '.mc16e'
        if kwargs['friend_directory']:
            self.attach_friend_files(kwargs['friend_directory'])
        self.trees_with_friends = None

    def open(self, file_name=None):
        if self.tfile is not None and self.tfile.IsOpen():
            return
        if file_name is not None:
            return TFile.Open(file_name, "READ")
        if self.path is not None and not os.path.exists(self.absFName) and "create" not in self.open_option.lower():
            raise ValueError("File " + os.path.join(self.path, self.file_name) + " does not exist.")
        if self.open_option.lower() == "update" and self.run_dir is not None:
            self.initial_file_name = self.file_name
            copy_dir = os.path.join(self.run_dir, self.file_name.split("/")[-2])
            make_dirs(copy_dir)
            self.file_name = os.path.join(copy_dir, self.file_name.split("/")[-1])
            move(self.initial_file_name, self.file_name)
            time.sleep(1)
            while not os.path.exists(self.file_name):
                time.sleep(1)
        _logger.debug("Opening file {:s}".format(self.absFName))
        self.tfile = TFile.Open(self.absFName, self.open_option)

    def __del__(self):
        if self.tfile is None:
            return
        _logger.log(0, "Delete file handle for {:s}".format(self.tfile.GetName()))
        self.close()

    def exists(self):
        if self.absFName is None:
            return False
        return os.path.exists(self.absFName)

    def close(self):
        if self.tfile is None or not self.tfile.IsOpen():
            return
        _logger.log(0, "Closing file {:s}".format(self.tfile.GetName()))
        self.tfile.Close()
        if self.initial_file_name is not None:
            move(self.file_name, self.initial_file_name)

    def get_directory(self, directory):
        self.open()
        if directory is None:
            return self.tfile
        try:
            try:
                return self.tfile.Get(bytes(directory, encoding='utf8'))
            except TypeError:
                # python3
                return self.tfile.Get(directory)
        except Exception as e:
            print(str(e))
            raise e

    def get_objects(self, tdirectory=None):
        self.open()
        objects = []
        tdir = self.tfile
        if tdirectory is not None:
            tdir = self.get_directory(tdirectory)
        for obj in tdir.GetListOfKeys():
            objects.append(tdir.Get(obj.GetName()))  # , encoding='utf8'
        return objects

    def get_objects_by_type(self, typename, tdirectory=None):
        obj = self.get_objects(tdirectory)
        obj = [t for t in obj if t.InheritsFrom(typename)]
        return obj

    def get_objects_by_pattern(self, pattern, tdirectory=None):
        tdir = self.get_directory(tdirectory)
        objects = []
        pattern = re.compile(pattern)
        for key in tdir.GetListOfKeys():
            if re.search(pattern, key.GetName()):
                objects.append(tdir.Get(key.GetName()))
        if len(objects) == 0:
            _logger.warning("Could not find objects matching %s in %s" % (pattern, tdir.GetName()))
        return objects

    def get_branch_names_from_tree(self, tree_name, tdirectory=None, pattern=".*"):
        tree = self.get_object_by_name(tree_name, tdirectory)
        pattern = re.compile(pattern)
        branch_names = []
        for branch in tree.GetListOfBranches():
            if re.search(pattern, branch.GetName()):
                branch_names.append(branch.GetName())
        return branch_names

    def get_object_by_name(self, obj_name, tdirectory=None, friend_file=None):
        self.open()
        if friend_file is None:
            tdir = self.tfile
        else:
            tdir = friend_file
        if tdirectory:
            try:
                if friend_file:
                    tdir = self.get_object_by_name(tdirectory, friend_file=friend_file)
                else:
                    tdir = self.get_object_by_name(tdirectory)
            except ValueError as e:
                raise e
        try:
            # python2 w/ future
            obj = tdir.Get(bytes(obj_name, encoding='utf8'))
        except TypeError:
            # python3
            obj = tdir.Get(obj_name)
        if not obj.__nonzero__():
            raise ValueError("Object {:s} does not exist in directory {:s} "
                             "in file {:s}".format(obj_name, str(tdirectory), os.path.join(self.path, self.file_name)))
        return obj

    def get_number_of_total_events(self, unweighted=False):
        try:
            cutflow_hist = self.get_object_by_name('cutflow_DxAOD', tdirectory='Nominal')
            if unweighted:
                return cutflow_hist.GetBinContent(1)
            return cutflow_hist.GetBinContent(2)
        except ValueError as e:
            _logger.error("Unable to parse cutflow Nominal/DxAOD from file %s" % self.file_name)
            raise e

    def get_daod_events(self):
        try:
            cutflow_hist = self.get_object_by_name("EventLoop_EventCount")
            return cutflow_hist.GetBinContent(1)
        except ValueError as e:
            _logger.error("Unable to parse EventLoop_EventCount from file %s" % self.file_name)
            raise e

    def fetch_and_link_hist_to_tree(self, tree_name, hist, var_name, cut_string="", tdirectory=None, weight=None):
        tree = self.get_object_by_name(tree_name, tdirectory)
        if self.friends is not None:
            self.link_friend_trees(tree, tdirectory)
        _logger.debug("Parsed tree %s from file %s containing %i entries" % (tree_name, self.file_name,
                                                                             tree.GetEntries()))
        return project_hist(tree, hist, var_name, cut_string, weight, self.is_data)

    def link_friend_trees(self, nominal_tree, tdirectory):
        if isinstance(nominal_tree, str):
            nominal_tree = self.get_object_by_name(nominal_tree, tdirectory)
        if self.friend_tree_names is None:
            _logger.error("No friend tree names provided, but requested to link them.")
            return
        for friend_file in self.friend_files:
            friend_trees = [t for t in
                            [self.get_object_by_name(tn, tdirectory, friend_file) for tn in self.friend_tree_names] if
                            t is not None]
            if self.trees_with_friends is None:
                self.trees_with_friends = []
            for tree in friend_trees:
                nominal_tree.AddFriend(tree)
                self.trees_with_friends.append((nominal_tree, tree))

    def reset_friends(self):
        if self.friends is None:
            return
        for f in self.friends:
            friend_file = TFile.Open(f, "READ")
            self.friend_files.append(friend_file)

    def release_friends(self):
        if self.trees_with_friends is None:
            return
        for tree, friend in self.trees_with_friends:
            tree.RemoveFriend(friend)
        self.trees_with_friends = None
        self.friend_files = []

    def attach_friend_files(self, directory):
        self.friends = []
        available_files = os.listdir(directory)
        base_file_name = self.file_name.split("/")[-1]
        for pattern in self.friend_pattern:
            try:
                friend_fn = [fn for fn in available_files
                             if fn == base_file_name.replace("ntuple", pattern).replace("hist", pattern)][0]
                self.friends.append(os.path.join(directory, friend_fn))
            except IndexError:
                _logger.error("Could not find friend for {:s}".format(base_file_name))

    @staticmethod
    def release_object_from_file(obj):
        obj.SetDirectory(0)
