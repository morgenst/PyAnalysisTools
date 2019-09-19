from __future__ import print_function

from copy import copy
from itertools import chain, combinations, permutations

from builtins import map
from builtins import object
from builtins import range
from builtins import zip

import PyAnalysisTools.PlottingUtils.Formatting as FM
import PyAnalysisTools.PlottingUtils.HistTools as HT
import PyAnalysisTools.PlottingUtils.PlottingTools as PT
import ROOT
from PyAnalysisTools.AnalysisTools.EfficiencyCalculator import EfficiencyCalculator as ec
from PyAnalysisTools.AnalysisTools.XSHandle import XSHandle
from PyAnalysisTools.PlottingUtils.BasePlotter import BasePlotter
from PyAnalysisTools.PlottingUtils.PlotConfig import PlotConfig as pc
from PyAnalysisTools.PlottingUtils.PlotConfig import find_process_config, get_histogram_definition
from PyAnalysisTools.PlottingUtils.Plotter import Plotter
from PyAnalysisTools.ROOTUtils.FileHandle import FileHandle
from PyAnalysisTools.ROOTUtils.ObjectHandle import find_branches_matching_pattern
from PyAnalysisTools.ROOTUtils.ObjectHandle import get_objects_from_canvas_by_type
from PyAnalysisTools.base import _logger, InvalidInputError
from PyAnalysisTools.base.OutputHandle import OutputFileHandle
from PyAnalysisTools.base.YAMLHandle import YAMLLoader as yl


class TriggerFlattener(object):
    def __init__(self, **kwargs):
        if not "input_file" in kwargs:
            raise InvalidInputError("No input file name provided")
        if not "tree_name" in kwargs:
            raise InvalidInputError("No tree name provided")
        kwargs.setdefault("additional_trees", [])
        kwargs.setdefault("tmp_dir", None)
        kwargs.setdefault("branch_name", "triggerList")
        self.file_handle = FileHandle(file_name=kwargs["input_file"], run_dir=kwargs["tmp_dir"], open_option="UPDATE")
        self.tree_name = kwargs["tree_name"]
        self.tree = self.file_handle.get_object_by_name(self.tree_name, tdirectory="Nominal")
        self.branch_name = kwargs["branch_name"]
        self.additional_trees_names = kwargs["additional_trees"]
        if self.additional_trees_names is None:
            self.additional_trees_names = []
        for tn in self.additional_trees_names:
            setattr(self, tn, self.file_handle.get_object_by_name(tn, tdirectory="Nominal"))
        self.trigger_list = []

    def flatten_all_branches(self, skipAcceptance=False):
        # branch_names = find_branches_matching_pattern(self.tree, "^trigger_.*")
        branch_names = find_branches_matching_pattern(self.tree, "^trigger.*")
        self.read_triggers()
        # branch_names.remove("trigger_list")
        branch_names.remove("triggerList")
        self.expand_branches(branch_names)

    def read_triggers(self):
        for entry in range(self.tree.GetEntries()):
            self.tree.GetEntry(entry)
            # for item in range(len(self.tree.trigger_list)):
            #     if self.tree.trigger_list[item].replace("-", "_") not in self.trigger_list:
            #         self.trigger_list.append(self.tree.trigger_list[item].replace("-", "_"))
            for item in range(len(self.tree.triggerList)):
                if self.tree.triggerList[item].replace("-", "_") not in self.trigger_list:
                    self.trigger_list.append(self.tree.triggerList[item].replace("-", "_"))

    def expand_branches(self, branch_names, skipAcceptance=False):
        for branch_name in branch_names:
            for trigger_name in self.trigger_list:
                new_name = branch_name.replace("trigger", trigger_name)
                if "acceptance" in new_name:
                    new_trigName = new_name
                    if skipAcceptance:
                        new_trigName = new_name.replace("_acceptance", "").replace("Acceptance", "")
                    exec ("data_holder_{:s} = array(\'i\', [0])".format(new_name))
                    exec ("branch_{:s} = self.tree.Branch(\"{:s}\", data_holder_{:s}, \"{:s}/I\")".format(new_name,
                                                                                                          new_trigName,
                                                                                                          new_name,
                                                                                                          new_trigName))
                    for tn in self.additional_trees_names:
                        exec ("branch_{:s}_{:s} = self.{:s}.Branch(\"{:s}\", data_holder_{:s}, \"{:s}/I\")".format(tn,
                                                                                                                   new_name,
                                                                                                                   tn,
                                                                                                                   new_trigName,
                                                                                                                   new_name,
                                                                                                                   new_trigName))
                else:
                    exec ("data_holder_{:s} = array(\'f\', [0.])".format(new_name))
                    exec ("branch_{:s} = self.tree.Branch(\"{:s}\", data_holder_{:s}, \"{:s}/F\")".format(
                        *[new_name] * 4))
                    for tn in self.additional_trees_names:
                        exec ("branch_{:s}_{:s} = self.{:s}.Branch(\"{:s}\", data_holder_{:s}, \"{:s}/F\")".format(tn,
                                                                                                                   new_name,
                                                                                                                   tn,
                                                                                                                   *[
                                                                                                                        new_name] * 3))

        for entry in range(self.tree.GetEntries()):
            self.tree.GetEntry(entry)
            for tree_name in self.additional_trees_names:
                getattr(self, tree_name).GetEntry(entry)
            unprocessed_triggers = copy(self.trigger_list)
            exec ("trig_list_branch = self.tree.{:s}".format(self.branch_name))
            # for item in range(len(self.tree.trigger_list)):
            #     trig_name = self.tree.trigger_list[item].replace("-", "_")
            for item in range(len(self.tree.triggerList)):
                trig_name = self.tree.triggerList[item].replace("-", "_")

                if trig_name not in unprocessed_triggers:
                    _logger.warning("{:s} not in unprocessed trigger list. Likely there went something wrong in the "
                                    "branch filling".format((trig_name)))
                    continue
                unprocessed_triggers.remove(trig_name)
                for branch_name in branch_names:
                    new_name = branch_name.replace("trigger", trig_name)
                    exec ("data_holder_{:s}[0] = self.tree.{:s}[item]".format(new_name, branch_name))
                    eval("branch_{:s}.Fill()".format(new_name))
                    for tn in self.additional_trees_names:
                        eval("branch_{:s}_{:s}.Fill()".format(tn, new_name))
            for missing_trigger in unprocessed_triggers:
                for branch_name in branch_names:
                    new_name = branch_name.replace("trigger", missing_trigger)
                    exec ("data_holder_{:s}[0] = -1111".format(new_name))
                    eval("branch_{:s}.Fill()".format(new_name))
                    for tn in self.additional_trees_names:
                        eval("branch_{:s}_{:s}.Fill()".format(tn, new_name))
        tdir = self.file_handle.get_directory("Nominal")
        tdir.cd()
        self.tree.Write()
        for tree_name in self.additional_trees_names:
            getattr(self, tree_name).Write()


class TriggerAcceptancePlotter(BasePlotter):
    def __init__(self, **kwargs):
        kwargs.setdefault("datset_info", None)
        kwargs.setdefault("output_file_name", "plots.root")
        self.file_handles = [FileHandle(file_name=file_name, dataset_info=kwargs["xs_config_file"])
                             for file_name in kwargs["input_files"]]
        self.tree_name = kwargs["tree_name"]
        self.hist_def = None
        super(TriggerAcceptancePlotter, self).__init__(**kwargs)
        self.xs_handle = XSHandle(kwargs["xs_config_file"])
        self.output_handle = OutputFileHandle(make_plotbook=self.plot_configs[0].make_plot_book, **kwargs)
        self.trigger_list = self.build_trigger_list()
        self.trigger_list = [t for t in self.trigger_list if 'prescale' not in t and 'online' not in t]
        self.overlap_hist = None
        self.unqiue_rate_hist = None

    def __del__(self):
        self.output_handle.write_and_close()

    def build_trigger_list(self):
        trigger_list = list(set(list(chain.from_iterable([file_handle.get_branch_names_from_tree(self.tree_name,
                                                                                                 tdirectory="Nominal",
                                                                                                 pattern="HLT_") for
                                                          file_handle in self.file_handles]))))
        if hasattr(self.plot_configs[0], 'white_list'):
            trigger_list = [x for x in trigger_list if x in self.plot_configs[0].white_list]
        if hasattr(self.plot_configs[0], 'black_list'):
            trigger_list = [x for x in trigger_list if x not in self.plot_configs[0].black_list]
        return trigger_list

    def get_hist_def(self, name):
        if self.hist_def is not None:
            return self.hist_def.Clone(name)
        hist = ROOT.TH1F(name, "", len(self.trigger_list), 0., len(self.trigger_list))
        for trigger_name in self.trigger_list:
            hist.GetXaxis().SetBinLabel(self.trigger_list.index(trigger_name) + 1,
                                        trigger_name.replace("_acceptance", "").replace("Acceptance", ""))
            hist.GetXaxis().SetLabelSize(0.03)
        return hist

    def apply_lumi_weights(self, histograms):
        for process, hist in list(histograms.items()):
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
            for process_name in list(histograms.keys()):
                _ = find_process_config(process_name, self.process_configs)
        Plotter.merge(histograms, self.process_configs)
        # canvas = PT.plot_stack(histograms, self.plot_configs[0]) -- mmm dev

        canvas = PT.plot_objects(histograms, self.plot_configs[0])
        canvas = FM.format_canvas(canvas, margin={"right": 0.15, "bottom": 0.2})
        canvas.Modified()
        FM.decorate_canvas(canvas, self.plot_configs[0])
        self.output_handle.register_object(canvas)

    def read_data(self, file_handle, trigger_data={}):
        tree = file_handle.get_object_by_name(self.tree_name, tdirectory="Nominal")
        entries = tree.GetEntries()
        for entry in range(entries):
            tree.GetEntry(entry)
            for trigger in self.trigger_list:
                try:
                    trigger_data[trigger].append(eval("tree.{:s}".format(trigger)))
                except KeyError:
                    trigger_data[trigger] = [(eval("tree.{:s}".format(trigger)))]
        return trigger_data

    def get_overlap_coefficients(self, trigger_data):
        trigger_combinations = list(combinations(self.trigger_list, 2))
        trigger_overlap = {}
        for comb in trigger_combinations:
            if sum(trigger_data[comb[0]]) > 0. and sum(trigger_data[comb[1]]) > 0.:
                overlap = sum(map(float, [d[0] == d[1] and d[0] == 1 for d in zip(trigger_data[comb[0]],
                                                                                       trigger_data[comb[1]])]))
                overlap /= sum(map(float, [d[0] == 1 or d[1] == 1 for d in zip(trigger_data[comb[0]],
                                                                                    trigger_data[comb[1]])]))
            else:
                overlap = 0.
            trigger_overlap[comb] = overlap
        return trigger_overlap

    def get_unique_correlation_coefficients(self, trigger_data):
        trigger_combinations = list(permutations(self.trigger_list, 2))
        trigger_overlap = {}
        for comb in trigger_combinations:
            if sum(trigger_data[comb[0]]) > 0. and sum(trigger_data[comb[1]]) > 0.:
                overlap = sum(map(float, [d[0] == d[1] and d[0] == 1 for d in zip(trigger_data[comb[0]],
                                                                                       trigger_data[comb[1]])]))
                try:
                    overlap /= sum(map(float, [d[0] == 1 for d in zip(trigger_data[comb[0]],
                                                                           trigger_data[comb[1]])]))
                except ZeroDivisionError:
                    overlap = 0.
            else:
                overlap = 0.
            trigger_overlap[comb] = overlap
        return trigger_overlap

    def get_unique_rate(self, trigger_data):
        trigger_unqiue_rate = {}
        print(trigger_data)
        exit(0)
        for comb in trigger_combinations:
            overlap = sum(map(float, [d[0] == d[1] and d[0] == 1 for d in zip(trigger_data[comb[0]],
                                                                                   trigger_data[comb[1]])]))
            overlap /= sum(map(float, [d[0] == 1 or d[1] == 1 for d in zip(trigger_data[comb[0]],
                                                                                trigger_data[comb[1]])]))
            trigger_overlap[comb] = overlap
        return trigger_overlap

    def plot_trigger_correlation(self):
        process_dict = {}
        for file_handle in self.file_handles:
            if file_handle.process in process_dict:
                process_dict[file_handle.process].append(file_handle)
            else:
                process_dict[file_handle.process] = [file_handle]

        for process, file_handles in list(process_dict.items()):
            trigger_data = {}
            for file_handle in file_handles:
                trigger_data = self.read_data(file_handle, trigger_data)
            overlap = self.get_overlap_coefficients(trigger_data)
            self.output_handle.register_object(self.make_overlap_histogram("overlap_{:s}".format(file_handle.process),
                                                                           overlap))

    def plot_trigger_unique_correlation(self):
        process_dict = {}
        for file_handle in self.file_handles:
            if file_handle.process in process_dict:
                process_dict[file_handle.process].append(file_handle)
            else:
                process_dict[file_handle.process] = [file_handle]

        for process, file_handles in list(process_dict.items()):
            trigger_data = {}
            for file_handle in file_handles:
                trigger_data = self.read_data(file_handle, trigger_data)
            overlap = self.get_unique_correlation_coefficients(trigger_data)
            self.output_handle.register_object(
                self.make_overlap_histogram("unique_correlation_{:s}".format(file_handle.process),
                                            overlap, unique=True))

    def plot_unqiue_trigger_rate(self):
        for file_handle in self.file_handles:
            trigger_data = self.read_data(file_handle)
            unique_rate = self.get_unique_rate(trigger_data)
            self.output_handle.register_object(
                self.make_unique_rate_histogram("unqiue_rate_{:s}".format(file_handle.process),
                                                unique_rate))

    def make_overlap_histogram(self, name, data, unique=False):
        ROOT.gStyle.SetPaintTextFormat("4.2f")

        def get_hist_def():
            self.overlap_hist = ROOT.TH2F(name, "", len(self.trigger_list), 0., len(self.trigger_list),
                                          len(self.trigger_list), 0., len(self.trigger_list))
            for trigger_name in self.trigger_list:
                index = self.trigger_list.index(trigger_name)
                self.overlap_hist.GetXaxis().SetBinLabel(index + 1,
                                                         trigger_name.replace("_acceptance", "").replace("Acceptance",
                                                                                                         ""))
                self.overlap_hist.GetXaxis().SetLabelSize(0.02)
                self.overlap_hist.GetYaxis().SetBinLabel(len(self.trigger_list) - index,
                                                         trigger_name.replace("_acceptance", "").replace("Acceptance",
                                                                                                         ""))
                self.overlap_hist.GetYaxis().SetLabelSize(0.02)
                self.overlap_hist.GetZaxis().SetLabelSize(0.03)

        get_hist_def()
        for comb, overlap in list(data.items()):
            self.overlap_hist.Fill(comb[0].replace("_acceptance", "").replace("Acceptance", ""),
                                   comb[1].replace("_acceptance", "").replace("Acceptance", ""), overlap)
            if not unique:
                self.overlap_hist.Fill(comb[1].replace("_acceptance", "").replace("Acceptance", ""),
                                       comb[0].replace("_acceptance", "").replace("Acceptance", ""), overlap)
        for i in range(self.overlap_hist.GetNbinsX()):
            self.overlap_hist.Fill(i, self.overlap_hist.GetNbinsX() - i - 1, 1.)
        ztitle = "(trigger0 || trigger1)/(trigger0 && trigger1)"
        if unique:
            ztitle = "(trigger0 && trigger1)/trigger0"
        plot_config = pc(name=name, draw_option="COLZTEXT",
                         xtitle="trigger 0", xtitle_offset=2.0, xtitle_size=0.04,
                         ytitle="trigger 1", ytitle_offset=2.8, ytitle_size=0.04,
                         ztitle=ztitle, ztitle_size=0.04)
        canvas = PT.plot_2d_hist(self.overlap_hist, plot_config=plot_config)
        canvas = FM.format_canvas(canvas, margin={"left": 0.2, "bottom": 0.16})
        canvas.Update()
        return canvas

    def make_unique_rate_histogram(self, name, data):
        def get_hist_def():
            if self.unique_rate_hist is not None:
                return self.unique_rate_hist.Clone(name)
            self.unique_rate_hist = ROOT.TH1F(name, "", len(self.trigger_list), 0., len(self.trigger_list))
            for trigger_name in self.trigger_list:
                index = self.trigger_list.index(trigger_name)
                self.unqiue_rate_hist.GetXaxis().SetBinLabel(index + 1, trigger_name.replace("_acceptance", "").replace(
                    "Acceptance", ""))
                self.unqiue_rate_hist.GetXaxis().SetLabelSize(0.03)

        get_hist_def()
        for comb, overlap in list(data.items()):
            self.overlap_hist.Fill(comb[0].replace("_acceptance", "").replace("Acceptance", ""),
                                   comb[1].replace("_acceptance", "").replace("Acceptance", ""), overlap)
            self.overlap_hist.Fill(comb[1].replace("_acceptance", "").replace("Acceptance", ""),
                                   comb[0].replace("_acceptance", "").replace("Acceptance", ""), overlap)
        plot_config = pc(name=name, draw_option="HIST")
        return PT.plot_hist(self.unqiue_rate_hist, plot_config=plot_config)

    def make_histograms(self, data):
        histograms = {}
        for process, trigger_info in list(data.items()):
            hist = self.get_hist_def("trigger_" + process)
            histograms[process] = self.fill_histogram(hist, trigger_info)
        return histograms

    @staticmethod
    def fill_histogram(hist, data):
        for label, count in list(data.items()):
            hist.Fill(label, count)
        return hist

    def read_triggers(self):
        def parse_trigger_info(file_handle):
            tree = file_handle.get_object_by_name(self.tree_name, tdirectory="Nominal")
            tmp = dict((trigger.replace("_acceptance", "").replace("Acceptance", ""),
                        tree.GetEntries("{:s} == 1".format(trigger))) for trigger in self.trigger_list)
            return tmp

        data = dict((file_handle.process, parse_trigger_info(file_handle)) for file_handle in self.file_handles)
        return data


class TriggerEfficiencyAnalyser(BasePlotter):
    def __init__(self, **kwargs):
        if not "tree_name" in kwargs:
            raise InvalidInputError("No tree name provided")
        self.file_list = kwargs["input_files"]
        self.tree_name = kwargs["tree_name"]
        self.file_handles = [FileHandle(file_name=file_name) for file_name in self.file_list]
        self.calculator = ec()
        super(TriggerEfficiencyAnalyser, self).__init__(**kwargs)
        self.output_handle = OutputFileHandle(**kwargs)
        self.config = yl.read_yaml(kwargs["config_file"])
        self.denominators = {}
        self.efficiencies = {}

    def get_denominators(self, plot_config):
        if plot_config.dist not in self.denominators:
            hist = get_histogram_definition(plot_config)
            cut_string = ""
            if hasattr(plot_config, "cut"):
                cut_string = plot_config.cut
            self.denominators[plot_config.dist] = dict((file_handle.process,
                                                        file_handle.fetch_and_link_hist_to_tree(self.tree_name,
                                                                                                hist,
                                                                                                plot_config.dist,
                                                                                                cut_string=cut_string,
                                                                                                tdirectory="Nominal"))
                                                       for file_handle in self.file_handles)
        return self.denominators[plot_config.dist]

    def get_efficiency(self, trigger_name, plot_config):
        denominators = self.get_denominators(plot_config)
        numerator_plot_config = copy(plot_config)
        numerator_plot_config.dist = numerator_plot_config.numerator.replace("replace", trigger_name)
        numerator_plot_config.name = numerator_plot_config.numerator.replace("replace", trigger_name).split()[0]

        hist = get_histogram_definition(numerator_plot_config)
        cut_string = ""
        if hasattr(plot_config, "cut"):
            cut_string = plot_config.cut.replace(plot_config.dist, numerator_plot_config.dist)
        numerators = dict((file_handle.process,
                           file_handle.fetch_and_link_hist_to_tree(self.tree_name,
                                                                   hist,
                                                                   numerator_plot_config.dist,
                                                                   cut_string=cut_string,
                                                                   tdirectory="Nominal")) for file_handle in
                          self.file_handles)
        if not isinstance(list(numerators.values())[0], ROOT.TH2F):
            dependency = plot_config.name.split("_")[-1]
        else:
            dependency = "QetavsPt"
        efficiencies = dict((process, self.calculator.calculate_efficiency(numerators[process],
                                                                           denominators[process],
                                                                           name="eff_{:s}_{:s}_{:s}".format(process,
                                                                                                            trigger_name,
                                                                                                            dependency)))
                            for process in list(numerators.keys()))
        if not isinstance(list(numerators.values())[0], ROOT.TH2F):
            canvas = PT.plot_objects(efficiencies, plot_config)
        else:
            plot_config.name = list(efficiencies.values())[0].GetName()
            canvas = PT.plot_obj(list(efficiencies.values())[0], plot_config)
        if "dr" in plot_config.name:
            self.fit_efficiency(canvas)
        return canvas

    def fit_efficiency(self, canvas):
        efficiency_graphs = get_objects_from_canvas_by_type(canvas, "TEfficiency")
        fit_func = ROOT.TF1("fermi", "[3] / ([0] + [3] *[4]) * ( [0] / (exp(-[1] * x + [2]) + [3])) + [4]", 0., 0.3)
        fit_func.SetParameters(1, 10, 0., 0., 0.)
        fit_func.SetParLimits(0, -5., 20.)
        fit_func.SetParLimits(1, -100., 100.)
        fit_func.SetParLimits(2, -10., 10.)
        fit_func.SetParLimits(3, -10., 10.)
        fit_func.SetParLimits(4, -10., 10.)
        # print efficiency_graphs
        # a = ROOT.RooRealVar("a", "", 1, 5., 20, )
        # b = ROOT.RooRealVar("b", "", 10., -100., 100.)
        # c = ROOT.RooRealVar("c", "", 0., -10., 10.)
        # d = ROOT.RooRealVar("d", "", 0., -10., 10.)
        # e = ROOT.RooRealVar("e", "", 0., -10., 10.)
        # dr = ROOT.RooRealVar("dr", "#Delta R", 0., 0.3)
        # fermi = ROOT.RooGenericPdf("fermi", "fermi function", "d / (a + d *e) * ( a / (exp(-B * dr + c) + d)) + e",
        # #ROOT.RooArgSet(dr, a, b, c, d, e))
        # data = ROOT.RooDataHist("efficiency", "", )
        for teff in efficiency_graphs:
            teff.Fit(fit_func)
            fit_func.Draw("L same")

    def get_efficiencies(self):
        for plot_config in self.plot_configs:
            for trigger_name in self.config["triggers"]:
                canvas = self.get_efficiency(trigger_name, plot_config)
                self.efficiencies[trigger_name] = canvas
                self.output_handle.register_object(canvas)
        self.output_handle.write_and_close()
