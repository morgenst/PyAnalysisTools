import os
import re
import time
from ROOT import TFile
from PyAnalysisTools.base import _logger, InvalidInputError
from PyAnalysisTools.base.YAMLHandle import YAMLLoader
from PyAnalysisTools.base.ShellUtils import resolve_path_from_symbolic_links, make_dirs, move


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

#@memoize
class FileHandle(object):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("path", "./")
        kwargs.setdefault("cwd", "None")
        kwargs.setdefault("open_option", "READ")
        kwargs.setdefault("run_dir", None)
        kwargs.setdefault("switch_off_process_name_analysis", False)
        self.file_name = resolve_path_from_symbolic_links(kwargs["cwd"], kwargs["file_name"])
        self.path = resolve_path_from_symbolic_links(kwargs["cwd"], kwargs["path"])
        self.absFName = os.path.join(self.path, self.file_name)
        #todo: inefficient as each file handle holds dataset_info. should be retrieved from linked store
        self.dataset_info = None
        if "dataset_info" in kwargs and kwargs["dataset_info"] is not None:
            self.dataset_info = YAMLLoader.read_yaml(kwargs["dataset_info"])
        self.open_option = kwargs["open_option"]
        self.tfile = None
        self.initial_file_name = None
        self.run_dir = kwargs["run_dir"]
        self.open()
        self.year = None
        self.period = None
        self.is_data = False
        self.is_mc = False
        if "ignore_process_name" not in kwargs:
            self.process = self.parse_process(kwargs["switch_off_process_name_analysis"])

    def open(self):
        if not os.path.exists(self.absFName):
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
                return None
                self.is_data = True
                return "Data"
        process_name = self.file_name.split("-")[-1].split(".")[0]
        if switch_off_analysis:
            return process_name
        process_name = re.sub(r"(\_\d)$", "", process_name)
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

    def get_object_by_name(self, obj_name, tdirectory=None):
        self.open()
        tdir = self.tfile
        if tdirectory:
            try:
                tdir = self.get_object_by_name(tdirectory)
            except ValueError as e:
                raise e
        obj = tdir.Get(obj_name)
        if not obj.__nonzero__():
            raise ValueError("Object " + obj_name + " does not exist in file " + os.path.join(self.path,
                                                                                              self.file_name))
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
        _logger.debug("Parsed tree %s from file %s containing %i entries" % (tree_name, self.file_name,
                                                                             tree.GetEntries()))
        if cut_string is None:
            cut_string = ""
        if weight:
            mc_weights = None
            if "MC:" in weight:
                weight = weight.split("*")
                mc_weights = filter(lambda w: "MC:" in w, weight)
                for mc_w in mc_weights:
                    weight.remove(mc_w)
                weight = "*".join(weight)
                if not self.is_data:
                    mc_weights = map(lambda mc_w: mc_w.replace("MC:", ""), mc_weights)
                    for mc_w in mc_weights:
                        weight += "* {:s}".format(mc_w)
            if cut_string == "":
                cut_string = weight
            else:
                cut_string = "%s * (%s)" % (weight, cut_string)
        n_selected_events = tree.Project(hist.GetName(), var_name, cut_string)
        _logger.debug("Selected %i events from tree %s for distribution %s and cut %s." % (n_selected_events,
                                                                                           tree_name,
                                                                                           var_name,
                                                                                           cut_string))
        if n_selected_events != hist.GetEntries():
            _logger.error("No of selected events does not match histogram entries. Probably FileHandle has been " +
                          "initialised after histogram definition has been received")
            raise RuntimeError("Inconsistency in TTree::Project")
        if n_selected_events == -1:
            _logger.error("Unable to project %s from tree %s with cut %s" % (var_name, tree_name, cut_string))
            raise RuntimeError("TTree::Project failed")
        return hist

    @staticmethod
    def release_object_from_file(obj):
        obj.SetDirectory(0)
