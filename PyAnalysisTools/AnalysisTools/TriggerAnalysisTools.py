import ROOT
from copy import copy
from array import array
from itertools import chain
import PyAnalysisTools.PlottingUtils.PlottingTools as PT
import PyAnalysisTools.PlottingUtils.HistTools as HT
import PyAnalysisTools.PlottingUtils.Formatting as FM
from PyAnalysisTools.base import _logger, InvalidInputError
from PyAnalysisTools.base.OutputHandle import OutputFileHandle
from PyAnalysisTools.AnalysisTools.XSHandle import XSHandle
from PyAnalysisTools.ROOTUtils.FileHandle import FileHandle
from PyAnalysisTools.ROOTUtils.ObjectHandle import find_branches_matching_pattern
from PyAnalysisTools.PlottingUtils.BasePlotter import BasePlotter
from PyAnalysisTools.PlottingUtils.Plotter import Plotter
from PyAnalysisTools.PlottingUtils.PlotConfig import find_process_config


class TriggerFlattener(object):
    def __init__(self, **kwargs):
        if not "input_file" in kwargs:
            raise InvalidInputError("No input file name provided")
        if not "tree_name" in kwargs:
            raise InvalidInputError("No tree name provided")
        self.file_handle = FileHandle(file_name=kwargs["input_file"], open_option="UPDATE")
        self.tree_name = kwargs["tree_name"]
        self.tree = self.file_handle.get_object_by_name(self.tree_name, tdirectory="Nominal")
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


class TriggerAcceptancePlotter(BasePlotter):
    def __init__(self, **kwargs):
        kwargs.setdefault("datset_info", None)
        kwargs.setdefault("output_file_name", "plots.root")
        self.file_handles = [FileHandle(file_name=file_name, dataset_info=kwargs["dataset_info"])
                             for file_name in kwargs["input_file_list"]]
        self.tree_name = "BaseSelection_trigger_Final"
        self.trigger_list = self.build_trigger_list()
        self.hist_def = None
        super(TriggerAcceptancePlotter, self).__init__(**kwargs)
        self.xs_handle = XSHandle(kwargs["dataset_info"])
        self.output_handle = OutputFileHandle(make_plotbook=self.plot_configs[0].make_plot_book, **kwargs)

    def build_trigger_list(self):
        trigger_list = list(set(list(chain.from_iterable([file_handle.get_branch_names_from_tree(self.tree_name,
                                                                                            tdirectory="Nominal",
                                                                                            pattern="HLT_") for file_handle in self.file_handles]))))
        return trigger_list

    def get_hist_def(self, name):
        if self.hist_def is not None:
            return self.hist_def.Clone(name)
        hist = ROOT.TH1F(name, "", len(self.trigger_list), 0., len(self.trigger_list))
        for trigger_name in self.trigger_list:
            hist.GetXaxis().SetBinLabel(self.trigger_list.index(trigger_name) + 1,
                                        trigger_name.replace("_acceptance", ""))
            hist.GetXaxis().SetLabelSize(0.03)
        return hist

    def apply_lumi_weights(self, histograms):
        for process, hist in histograms.iteritems():
            if hist is None:
                _logger.error("Histogram for process {:s} is None".format(process))
                continue
            if "data" in process.lower():
                continue
            cross_section_weight = self.xs_handle.get_lumi_scale_factor(process, self.lumi, self.event_yields[process])
            HT.scale(hist, cross_section_weight)

    def plot_trigger_acceptance(self):
        self.read_cutflows()
        raw_data = self.read_triggers()
        histograms = self.make_histograms(raw_data)
        self.apply_lumi_weights(histograms)
        if self.process_configs is not None:
            for process_name in histograms.keys():
                _ = find_process_config(process_name, self.process_configs)
        Plotter.merge(histograms, self.process_configs)
        canvas = PT.plot_stack(histograms, self.plot_configs[0])
        canvas.SetBottomMargin(0.2)
        canvas.Modified()
        FM.decorate_canvas(canvas, self.plot_configs[0])
        self.output_handle.register_object(canvas)
        self.output_handle.write_and_close()

    def make_histograms(self, data):
        histograms = {}
        for process, trigger_info in data.iteritems():
            hist = self.get_hist_def("trigger_" + process)
            histograms[process] = self.fill_histogram(hist, trigger_info)
        return histograms

    @staticmethod
    def fill_histogram(hist, data):
        for label, count in data.iteritems():
            hist.Fill(label, count)
        return hist

    def read_triggers(self):
        def parse_trigger_info(file_handle):
            tree = file_handle.get_object_by_name(self.tree_name, tdirectory="Nominal")
            tmp = dict((trigger.replace("_acceptance", ""),
                        tree.GetEntries("{:s} == 1". format(trigger))) for trigger in self.trigger_list)
            return tmp
        data = dict((file_handle.process, parse_trigger_info(file_handle)) for file_handle in self.file_handles)
        return data
