import os
import re
import time
import ROOT
from ROOT import TFile
from PyAnalysisTools.base import _logger, InvalidInputError
from PyAnalysisTools.base.YAMLHandle import YAMLLoader
from PyAnalysisTools.base.ShellUtils import resolve_path_from_symbolic_links, make_dirs, move
from PyAnalysisTools.AnalysisTools.XSHandle import DataSetStore
from PyAnalysisTools.PlottingUtils.PlottingTools import project_hist

_memoized = {}


def get_id_tuple(f, args, kwargs, mark=object()):
    l = [hash(f)]
    for arg in args:
        l.append(hash(arg))
    l.append(id(mark))
    for k, v in kwargs.iteritems():
        l.append(k)
        l.append(id(v))
    return tuple(l)


def memoize(f):
    """
    Some basic memoizer
    """
    def memoized(*args, **kwargs):
        key = get_id_tuple(f, args, kwargs)
        if key not in _memoized:
            _memoized[key] = f(*args, **kwargs)
        return _memoized[key]
    return memoized


class FileHandle(object):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("path", "./")
        kwargs.setdefault("cwd", "None")
        kwargs.setdefault("open_option", "READ")
        kwargs.setdefault("run_dir", None)
        kwargs.setdefault("switch_off_process_name_analysis", False)
        kwargs.setdefault("friend_directory", None)
        kwargs.setdefault("friend_pattern", None)
        kwargs.setdefault("friend_tree_names", None)
        kwargs.setdefault("split_mc", False)
        kwargs.setdefault("dataset_info", None)
        self.file_name = resolve_path_from_symbolic_links(kwargs["cwd"], kwargs["file_name"])
        self.path = resolve_path_from_symbolic_links(kwargs["cwd"], kwargs["path"])
        self.absFName = os.path.join(self.path, self.file_name)
        self.dataset_info = None
        if "dataset_info" in kwargs and kwargs["dataset_info"] is not None:
            self.dataset_info = DataSetStore(kwargs["dataset_info"]).dataset_info
        self.open_option = kwargs["open_option"]
        self.tfile = None
        self.initial_file_name = None
        self.run_dir = kwargs["run_dir"]
        self.open()
        self.year = None
        self.period = None
        self.is_data = False
        self.is_mc = False
        self.mc16a = False
        self.mc16c = False
        self.mc16d = False
        self.mc16e = False
        self.mc_campaign = None
        self.friends = None
        self.friend_tree_names = kwargs["friend_tree_names"]
        self.friend_pattern = kwargs["friend_pattern"]
        self.friend_files = []
        if self.friend_tree_names is not None and not isinstance(self.friend_tree_names, list):
            self.friend_tree_names = [self.friend_tree_names]
        if self.friend_pattern is not None and not isinstance(self.friend_pattern, list):
            self.friend_pattern = [self.friend_pattern]
        if "ignore_process_name" not in kwargs:
            self.process = self.parse_process(kwargs["switch_off_process_name_analysis"])
            self.process_with_mc_campaign = self.process
            if self.process is not None:
                if self.mc16a:
                    self.process_with_mc_campaign += ".mc16a"
                if self.mc16c:
                    self.process_with_mc_campaign += ".mc16c"
                if self.mc16d:
                    self.process_with_mc_campaign += ".mc16d"
                if self.mc16e:
                    self.process_with_mc_campaign += ".mc16e"
            if kwargs["split_mc"]:
                self.process = self.process_with_mc_campaign
        if kwargs["friend_directory"]:
            self.attach_friend_files(kwargs["friend_directory"])
        self.trees_with_friends = None

    def open(self, file_name=None):
        if file_name is not None:
            return TFile.Open(file_name, "READ")
        if not os.path.exists(self.absFName) and "create" not in self.open_option.lower():
            raise ValueError("File " + os.path.join(self.path, self.file_name) + " does not exist.")
        if self.tfile is not None and self.tfile.IsOpen():
            return
        if self.open_option.lower() == "update" and self.run_dir is not None:
            self.initial_file_name = self.file_name
            copy_dir = os.path.join(self.run_dir, self.file_name.split("/")[-2])
            make_dirs(copy_dir)
            self.file_name = os.path.join(copy_dir, self.file_name.split("/")[-1])
            move(self.initial_file_name, self.file_name)
            time.sleep(1)
            self.absFName = os.path.join(self.path, self.file_name)
            while not os.path.exists(self.file_name):
                time.sleep(1)
        self.tfile = TFile.Open(os.path.join(self.path, self.file_name), self.open_option)

    def __del__(self):
        self.close()

    def close(self):
        if self.tfile is None or not self.tfile.IsOpen():
            return
        self.tfile.Close()
        if self.initial_file_name is not None:
            move(self.file_name, self.initial_file_name)

    def parse_process(self, switch_off_analysis=False):
        def analyse_process_name():
            if "user.shanisch" in process_name:
                self.year = process_name.split(".")[2]
                self.period = "periodB"
                return ".".join([self.year, self.period])    
            if "data" in process_name:
                try:
                    self.year, _, self.period = process_name.split("_")[0:3]
                    self.is_data = True
                    return ".".join([self.year, self.period])
                except ValueError:
                    _logger.warning("Unable to parse year and period from sample name {:s}".format(process_name))
                    return "Data"
            if self.dataset_info is not None:
                try:
                    tmp = filter(lambda l: l.dsid == int(process_name), self.dataset_info.values())
                except ValueError:
                    tmp = filter(lambda l: hasattr(l, "process_name") and l.process_name == process_name,
                                 self.dataset_info.values())
                if len(tmp) == 1:
                    self.mc = True
                    return tmp[0].process_name
            if process_name.isdigit():
                print "Could not find config for ", process_name
                return None
                # self.is_data = True
                # return "Data"

        if "mc16a" in self.file_name.lower():
            self.mc16a = True
            self.mc_campaign = 'mc16a'
        if "mc16c" in self.file_name.lower():
            self.mc16c = True
            self.mc_campaign = 'mc16c'
        if "mc16d" in self.file_name.lower():
            self.mc16d = True
            self.mc_campaign = 'mc16d'
        if "mc16e" in self.file_name.lower():
            self.mc16e = True
            self.mc_campaign = 'mc16e'
        process_name = self.file_name.split("-")[-1].split(".")[0]
        if switch_off_analysis:
            return process_name
        process_name = re.sub(r"(\_\d+)$", "", process_name)
        analysed_process_name = analyse_process_name()
        if analysed_process_name is None:
            process_name = self.file_name.split("/")[-2]
            analysed_process_name = analyse_process_name()
        return analysed_process_name

    def get_directory(self, directory):
        if directory is None:
            return self.tfile
        try:
            return self.tfile.Get(directory)
        except Exception as e:
            print e.msg()

    def get_objects(self, tdirectory=None):
        objects = []
        tdir = self.tfile
        if tdirectory is not None:
            tdir = self.get_directory(tdirectory)
        for obj in tdir.GetListOfKeys():
            objects.append(tdir.Get(obj.GetName()))
        return objects

    def get_objects_by_type(self, typename, tdirectory=None):
        obj = self.get_objects(tdirectory)
        obj = filter(lambda t: t.InheritsFrom(typename), obj)
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
        obj = tdir.Get(obj_name)
        if not obj.__nonzero__():
            raise ValueError("Object {:s} does not exist in directory {:s} in file {:s}".format(obj_name,
                                                                                                tdirectory,
                                                                                                os.path.join(self.path,
                                                                                                             self.file_name)))
        return obj

    def get_number_of_total_events(self, unweighted=False):
        try:
            cutflow_hist = self.get_object_by_name("Nominal/cutflow_DxAOD")
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
        if self.friend_tree_names is None:
            _logger.error("No friend tree names provided, but requested to link them.")
            return
        for friend_file in self.friend_files:
            friend_trees = filter(lambda t: t is not None,
                                  [self.get_object_by_name(tn, tdirectory, friend_file) for tn in self.friend_tree_names])
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
            friend_fn = filter(lambda fn: fn == base_file_name.replace("ntuple", pattern).replace("hist", pattern),
                              available_files)[0]
            self.friends.append(os.path.join(directory, friend_fn))

    @staticmethod
    def release_object_from_file(obj):
        obj.SetDirectory(0)
