__author__ = 'marcusmorgenstern'
__mail__ = ''

import os
from ROOT import TFile
from PyAnalysisTools.base import _logger, InvalidInputError

_memoized = {}


def get_id_tuple(f, args, kwargs, mark=object()):
    l = [id(f)]
    for arg in args:
        l.append(id(arg))
    l.append(id(mark))
    for k, v in kwargs:
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

@memoize
class FileHandle(object):
    def __init__(self, file_name, path='./'):
        self.file_name = file_name
        self.path = path
        self.absFName = os.path.join(self.path, self.file_name)
        self.open()
        self.process = self.parse_process()

    def open(self):
        if not os.path.exists(self.absFName):
            raise ValueError("File " + os.path.join(self.path, self.file_name) + " does not exist.")

        self.tfile = TFile.Open(os.path.join(self.path, self.file_name), 'READ')

    def parse_process(self):
        return self.file_name.split("-")[-1].split(".")[0]

    def get_objects(self):
        objects = []
        for obj in self.tfile.GetListOfKeys():
            objects.append(self.tfile.Get(obj.GetName()))
        return objects

    def get_objects_by_type(self, typename):
        obj = self.get_objects()
        obj = filter(lambda t: t.InheritsFrom(typename), obj)
        return obj

    def get_object_by_name(self, obj_name):
        obj = self.tfile.Get(obj_name)
        if not obj.__nonzero__():
            raise ValueError("Object " + obj_name + " does not exist in file " + os.path.join(self.path, self.file_name))
        #self.release_object_from_file(obj)
        return obj

    def fetch_and_link_hist_to_tree(self, tree_name, hist, var_name, cut_string=""):
        tree = self.get_object_by_name(tree_name)
        _logger.debug("Parsed tree %s from file %s containing %i entries" % (tree_name, self.file_name, tree.GetEntries()))
        if cut_string is None:
            cut_string = ""
        n_selected_events = tree.Project(hist.GetName(), var_name, cut_string)
        _logger.debug("Selected %i events from tree %s for distribution %s and cut %s." %(n_selected_events,
                                                                                          tree_name,
                                                                                          var_name,
                                                                                          cut_string))
        if n_selected_events == -1:
            _logger.error("Unable to project %s from tree %s with cut %s" % (var_name, tree_name, cut_string))
            raise RuntimeError("TTree::Project failed")
        return hist

    @staticmethod
    def release_object_from_file(obj):
        obj.SetDirectory(0)
