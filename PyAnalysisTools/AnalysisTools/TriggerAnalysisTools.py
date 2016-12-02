import ROOT
from copy import copy
from array import array
from PyAnalysisTools.base import _logger, InvalidInputError
from PyAnalysisTools.ROOTUtils.FileHandle import FileHandle
from PyAnalysisTools.ROOTUtils.ObjectHandle import find_branches_matching_pattern


class TriggerFlattener(object):
    def __init__(self, **kwargs):
        if not "input_file" in kwargs:
            raise InvalidInputError("No input file name provided")
        if not "tree_name" in kwargs:
            raise InvalidInputError("No tree name provided")
        self.file_handle = FileHandle(file_name=kwargs["input_file"], open_option="UPDATE")
        self.tree_name = kwargs["tree_name"]
        self.tree = self.file_handle.get_object_by_name(self.tree_name)
        self.trigger_list = []

    def flatten_all_branches(self):
        branch_names = find_branches_matching_pattern(self.tree, "trigger_*")
        self.read_triggers()
        branch_names.remove("trigger_list")
        self.expand_branches(branch_names)

    def read_triggers(self):
        for entry in range(self.tree.GetEntries()):
            self.tree.GetEntry(entry)
            for item in range(len(self.tree.trigger_list)):
                if self.tree.trigger_list[item].replace("-", "_") not in self.trigger_list:
                    self.trigger_list.append(self.tree.trigger_list[item].replace("-", "_"))

    def expand_branches(self, branch_names):
        for branch_name in branch_names:
            for trigger_name in self.trigger_list:
                new_name = branch_name.replace("trigger", trigger_name)
                exec("data_holder_{:s} = array(\'f\', [0.])".format(new_name))
                exec("branch_{:s} = self.tree.Branch(\"{:s}\", data_holder_{:s}, \"{:s}/F\")".format(*[new_name]*4))
        for entry in range(self.tree.GetEntries()):
            self.tree.GetEntry(entry)
            unprocessed_triggers = copy(self.trigger_list)
            for item in range(len(self.tree.trigger_list)):
                trig_name = self.tree.trigger_list[item].replace("-", "_")
                if trig_name not in unprocessed_triggers:
                    _logger.warning("{:s} not in unprocessed trigger list. Likely there went something wrong in the "
                                    "branch filling".format((trig_name)))
                    continue
                unprocessed_triggers.remove(trig_name)
                for branch_name in branch_names:
                    new_name = branch_name.replace("trigger", trig_name)
                    exec("data_holder_{:s}[0] = self.tree.{:s}[item]".format(new_name, branch_name))
                    eval("branch_{:s}.Fill()".format(new_name))
            for missing_trigger in unprocessed_triggers:
                for branch_name in branch_names:
                    new_name = branch_name.replace("trigger", missing_trigger)
                    exec ("data_holder_{:s}[0] = -1111.".format(new_name))
                    eval("branch_{:s}.Fill()".format(new_name))
        tdir = self.file_handle.get_directory("Nominal")
        tdir.cd()
        self.tree.Write()
