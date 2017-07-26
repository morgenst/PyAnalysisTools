import ROOT
import PyAnalysisTools.PlottingUtils.PlottingTools as PT
import PyAnalysisTools.PlottingUtils.Formatting as FM
from copy import copy
from PyAnalysisTools.PlottingUtils import set_batch_mode
from PyAnalysisTools.PlottingUtils.BasePlotter import BasePlotter
from PyAnalysisTools.base import _logger, InvalidInputError
from PyAnalysisTools.ROOTUtils.FileHandle import FileHandle
from PyAnalysisTools.base.OutputHandle import OutputFileHandle
from PyAnalysisTools.PlottingUtils.PlotConfig import get_histogram_definition, \
    expand_plot_config, parse_and_build_process_config, find_process_config, ProcessConfig
from PyAnalysisTools.ROOTUtils.ObjectHandle import get_objects_from_canvas_by_name, get_objects_from_canvas_by_type
from PyAnalysisTools.PlottingUtils.RatioPlotter import RatioPlotter
from PyAnalysisTools.AnalysisTools.SubtractionHandle import SubtractionHandle


class ComparisonReader(object):
    def __init__(self, **kwargs):
        if "input_files" not in kwargs:
            _logger.error("No input file provided")
            raise InvalidInputError("Missing input files")
        kwargs.setdefault("reference_files", None)
        kwargs.setdefault("reference_dataset_info", None)
        kwargs.setdefault("xs_config_file", None)
        self.input_files = kwargs["input_files"]
        self.reference_files = kwargs["reference_files"]
        self.tree_name = kwargs["tree_name"]
        for opt, val in kwargs.iteritems():
            if not hasattr(self, opt):
                setattr(self, opt, val)
        if hasattr(self, "merge_file") and not hasattr(self, "reference_merge_file"):
            self.reference_merge_file = self.merge_file
        if hasattr(self, "merge_file"):
            self.process_configs = self.parse_process_config(self.merge_file)
            self.reference_process_configs = self.parse_process_config(self.reference_merge_file)

    def get_instance(self, plot_config):
        if hasattr(plot_config, "dist") and hasattr(plot_config, "dist_ref") and len(self.input_files) == 1:
            _logger.debug("Using SingleFileMultiReader instance")
            return SingleFileMultiDistReader(input_files=self.input_files, plot_config=plot_config, tree_name=self.tree_name)
        if hasattr(plot_config, "dist") and not hasattr(plot_config, "dist_ref") and self.reference_files:
            _logger.debug("Using MultiFileSingleDistReader instance")
            return MultiFileSingleDistReader(plot_config=plot_config, **self.__dict__)
        if hasattr(plot_config, "dist") and hasattr(plot_config, "dist_ref") and self.reference_files is None and \
                        len(self.input_files) > 0:
            _logger.debug("Using MultiFileMultiDistReader")
            return MultiFileMultiDistReader(plot_config=plot_config, **self.__dict__)

    def get_data(self):
        data = {}
        for plot_config in self.plot_configs:
            getter = self.get_instance(plot_config)
            data[plot_config] = getter.get_data()
        return data

    def make_plot(self, file_handle, plot_config, tree_name=None):
        hist = get_histogram_definition(plot_config)
        hist.SetName(hist.GetName() + file_handle.process)
        if tree_name is None:
            tree_name = self.tree_name
        try:
            file_handle.fetch_and_link_hist_to_tree(tree_name, hist, plot_config.dist, None,
                                                    tdirectory=self.systematics)
            hist.SetName(hist.GetName() + "_" + file_handle.process)
            _logger.debug("try to access config for process %s" % file_handle.process)
        except Exception as e:
            raise e
        return hist

    @staticmethod
    def merge_histograms(histograms, process_configs):
        def expand():
            if process_configs is not None:
                for process_name in histograms.keys():
                    _ = find_process_config(process_name, process_configs)
        expand()
        for process, process_config in process_configs.iteritems():
            if not hasattr(process_config, "subprocesses"):
                continue
            for sub_process in process_config.subprocesses:
                if sub_process not in histograms.keys():
                    continue
                if process not in histograms.keys():
                    new_hist_name = histograms[sub_process].GetName().replace(sub_process, process)
                    histograms[process] = histograms[sub_process].Clone(new_hist_name)
                else:
                    histograms[process].Add(histograms[sub_process])
                histograms.pop(sub_process)

        for process in histograms.keys():
            histograms[find_process_config(process, process_configs)] = histograms.pop(process)


    @staticmethod
    def parse_process_config(process_config_file):
        if process_config_file is None:
            return None
        process_config = parse_and_build_process_config(process_config_file)
        return process_config


class SingleFileMultiDistReader(ComparisonReader):
    def __init__(self, **kwargs):
        input_files = kwargs["input_files"]
        plot_config = kwargs["plot_config"]
        if isinstance(input_files, list):
            if len(input_files) > 1:
                _logger.error("Privided {:d} input files for single file reader. "
                              "Using just first".format(len(input_files)))
            input_files = input_files[0]
        self.file_handle = FileHandle(file_name=input_files, switch_off_process_name_analysis=True)
        self.plot_config = plot_config
        self.tree_name = kwargs["tree_name"]

    def get_data(self):
        try:
            reference_canvas = self.file_handle.get_object_by_name(self.plot_config.dist_ref)
            compare_canvas = self.file_handle.get_object_by_name(self.plot_config.dist)
            if hasattr(self.plot_config, "retrieve_by") and self.plot_config.retrieve_by == "type":
                reference = get_objects_from_canvas_by_type(reference_canvas, "TH1F")
                compare = get_objects_from_canvas_by_type(compare_canvas, "TH1F")
            else:
                reference = get_objects_from_canvas_by_name(reference_canvas, self.plot_config.processes[0])
                compare = get_objects_from_canvas_by_name(compare_canvas, self.plot_config.processes[0])
        except ValueError:
            plot_configs = expand_plot_config(self.plot_config)
            plot_config_ref = copy(plot_configs[0])
            plot_config_ref.name += "_reference"
            plot_config_ref.dist = self.plot_config.dist_ref
            reference = self.make_plot(self.file_handle, plot_config_ref)
            compare = [self.make_plot(self.file_handle, plot_config) for plot_config in plot_configs]
            reference.SetDirectory(0)
            map(lambda obj: obj.SetDirectory(0), compare)
        return reference, compare


class SingleFileMultiProcessReader(ComparisonReader):
    def __init__(self):
        pass

    def get_data(self):
        pass


class MultiFileMultiDistReader(ComparisonReader):
    def __init__(self, **kwargs):
        self.file_handles = [FileHandle(file_name=fn,
                                        dataset_info=kwargs["xs_config_file"],
                                        switch_off_process_name_analysis=kwargs["xs_config_file"] is None)
                             for fn in kwargs["input_files"]]
        self.plot_config = kwargs["plot_config"]
        self.tree_name = kwargs["tree_name"]
        for opt, value in kwargs.iteritems():
            if not hasattr(self, opt):
                setattr(self, opt, value)

    def get_data(self):
        plot_config_ref = copy(self.plot_config)
        plot_config_ref.dist = self.plot_config.dist_ref
        plot_config_ref.name += "_ref"
        reference = {file_handle.process: self.make_plot(file_handle, plot_config_ref, self.tree_name) for
                     file_handle in self.file_handles}
        compare = {file_handle.process: self.make_plot(file_handle, self.plot_config, self.tree_name)
                   for file_handle in self.file_handles}
        ComparisonReader.merge_histograms(reference, self.process_configs)
        ComparisonReader.merge_histograms(compare, self.process_configs)
        if isinstance(reference, dict):
            map(lambda obj: obj.SetDirectory(0), reference.values())
            map(lambda obj: obj.SetDirectory(0), compare.values())
        else:
            map(lambda obj: obj.SetDirectory(0), reference)
            map(lambda obj: obj.SetDirectory(0), compare)
        if hasattr(plot_config_ref, "process_ref"):
            reference = dict(filter(lambda item: item[0].name in plot_config_ref.process_ref, reference.iteritems()))
        if hasattr(self.plot_config, "process_comp"):
            compare = dict(filter(lambda item: item[0].name in self.plot_config.process_comp, compare.iteritems()))

        return reference, compare


class MultiFileSingleDistReader(ComparisonReader):
    """
    Read same distribitionS from reference and input files
    """
    def __init__(self, **kwargs):
        reference_files = kwargs["reference_files"]
        input_files = kwargs["input_files"]
        plot_config = kwargs["plot_config"]
        self.reference_file_handles = [FileHandle(file_name=fn,
                                                  dataset_info=kwargs["reference_dataset_info"],
                                                  switch_off_process_name_analysis=kwargs["xs_config_file"] is None)
                                       for fn in reference_files]
        self.file_handles = [FileHandle(file_name=fn,
                                        dataset_info=kwargs["xs_config_file"],
                                        switch_off_process_name_analysis=kwargs["xs_config_file"] is None)
                             for fn in input_files]
        self.plot_config = plot_config
        self.tree_name = kwargs["tree_name"]
        self.reference_tree_name = copy(self.tree_name)
        if "reference_tree_name" in kwargs and not kwargs["reference_tree_name"] is None:
            self.reference_tree_name = kwargs["reference_tree_name"]
        for opt, value in kwargs.iteritems():
            if not hasattr(self, opt):
                setattr(self, opt, value)

    def get_data(self):
        try:
            reference_canvases = [file_handle.get_object_by_name(self.plot_config.dist)
                                  for file_handle in self.reference_file_handles]
            #todo: generalise to arbitrary number of compare inputs
            #todo: generalise to type given by plot config
            compare_canvases = [file_handle.get_object_by_name(self.plot_config.dist)
                              for file_handle in self.file_handles]
            obj_type = "TH1F"
            if hasattr(self.plot_config, "retrieve_by") and "type" in self.plot_config.retrieve_by:
                obj_type = self.plot_config.retrieve_by.replace("type:", "")
            reference = [get_objects_from_canvas_by_type(reference_canvas, obj_type)[0]
                         for reference_canvas in reference_canvases]
            compare = [get_objects_from_canvas_by_type(compare_canvas, obj_type)[0]
                       for compare_canvas in compare_canvases]
        except ValueError:
            plot_config_ref = copy(self.plot_config)
            plot_config_ref.name += "_reference"
            reference = {file_handle.process: self.make_plot(file_handle, plot_config_ref, self.reference_tree_name) for
                         file_handle in self.reference_file_handles}
            compare = {file_handle.process: self.make_plot(file_handle, self.plot_config)
                       for file_handle in self.file_handles}

            if hasattr(self, "process_configs") and self.process_configs is not None:
                ComparisonReader.merge_histograms(reference, self.reference_process_configs)
                ComparisonReader.merge_histograms(compare, self.process_configs)
            if isinstance(reference, dict):
                map(lambda obj: obj.SetDirectory(0), reference.values())
                map(lambda obj: obj.SetDirectory(0), compare.values())
            else:
                map(lambda obj: obj.SetDirectory(0), reference)
                map(lambda obj: obj.SetDirectory(0), compare)
        return reference, compare


class ComparisonPlotter(BasePlotter):
    def __init__(self, **kwargs):
        if not "input_files" in kwargs:
            _logger.error("No input files provided")
            raise InvalidInputError("Missing input files")
        if not "plot_config_files" in kwargs:
            _logger.error("No config file provided")
            raise InvalidInputError("Missing config")
        if not "output_dir" in kwargs:
            _logger.warning("No output directory given. Using ./")
        kwargs.setdefault("batch", True)
        kwargs.setdefault("tree_name", None)
        kwargs.setdefault("output_dir", "./")
        kwargs.setdefault("process_config_file", None)
        kwargs.setdefault("systematics", None)
        kwargs.setdefault("ref_mod_modules", None)
        kwargs.setdefault("inp_mod_modules", None)
        set_batch_mode(kwargs["batch"])
        super(ComparisonPlotter, self).__init__(**kwargs)
        self.input_files = kwargs["input_files"]
        self.output_handle = OutputFileHandle(overload="comparison", output_file_name="Compare.root", **kwargs)
        self.color_palette = [ROOT.kRed, ROOT.kBlue, ROOT.kGreen, ROOT.kCyan, ROOT.kPink, ROOT.kOrange, ROOT.kBlue-4,
                              ROOT.kRed+3, ROOT.kGreen-2]
        for attr, value in kwargs.iteritems():
            if not hasattr(self, attr):
                setattr(self, attr, value)
        if self.systematics is None:
            self.systematics = "Nominal"
        self.ref_process_configs = None
        if "reference_merge_file" in kwargs:
            self.ref_process_configs = parse_and_build_process_config(kwargs["reference_merge_file"])
        if "merge_file" in kwargs:
            self.process_config = parse_and_build_process_config(kwargs["reference_merge_file"])
        self.ref_mod_modules = [SubtractionHandle(subtract_items=["Wtaunu"], output_name="WmunuData",
                                                  reference_item="Data", process_configs=self.ref_process_configs)]
        self.analyse_plot_config()
        self.getter = ComparisonReader(plot_configs=self.plot_configs, **kwargs)


    def analyse_plot_config(self):
        pc = next((pc for pc in self.plot_configs if pc.name == "parse_from_file"), None)
        if pc is None:
            return
        if not hasattr(self, "reference_files"):
            _logger.error("Request to parse plot configs from file, but no reference file given. Breaking up!")
            exit(0)
        file_handles = [FileHandle(file_name=reference_file) for reference_file in self.reference_files]
        objects = []
        for file_handle in file_handles:
            objects += file_handle.get_objects_by_type("TCanvas")
        self.plot_configs.remove(pc)
        for obj in objects:
            new_pc = copy(pc)
            new_pc.dist = obj.GetName()
            self.plot_configs.append(new_pc)

    def update_color_palette(self):
        if isinstance(self.common_config.colors[0], str):
            self.color_palette = [getattr(ROOT, "k" + color.capitalize()) for color in self.common_config.colors]
        elif isinstance(self.common_config.colors[0], int):
            self.color_palette = [color for color in self.common_config.colors]
        else:
            _logger.warning("Unsuppored type %s for colors in common_config" % type(self.common_config.colors[0]))

    def make_comparison_plots(self):
        data = self.getter.get_data()
        for plot_config, hists in data.iteritems():
            self.make_comparison_plot(plot_config, hists)
        self.output_handle.write_and_close()

    def make_comparison_plot(self, plot_config, data):
        reference_hists = data[0]
        hists = data[1]
        labels = None
        if self.ref_mod_modules:
            for mod in self.ref_mod_modules:
                reference_hists = mod.execute(reference_hists)
        if isinstance(reference_hists, dict):
            reference_hists_dict = copy(reference_hists)
            hists_dict = copy(hists)
            reference_hists = reference_hists.values()
            hists = hists.values()
            labels = reference_hists_dict.keys() + hists_dict.keys()
            if any(isinstance(l, ProcessConfig) for l in labels):
                labels = map(lambda p: p.label, labels)
        y_max = None
        if not any([isinstance(hist, ROOT.TEfficiency) for hist in reference_hists]):
            yscale = 1.3
            ymax = yscale * max([item.GetMaximum() for item in hists] + [item.GetMaximum() for item in reference_hists])
            plot_config.yscale = yscale
            if not hasattr(plot_config, "normalise"):
                plot_config.ymax = ymax
        else:
            ctmp = ROOT.TCanvas("ctmp", "ctmp")
            ctmp.cd()
            reference_hists[0].Draw("ap")
            ROOT.gPad.Update()
            reference_hists[0].GetPaintedGraph().GetXaxis().GetTitle()
            plot_config.xtitle = reference_hists[0].GetPaintedGraph().GetXaxis().GetTitle()
            plot_config.ytitle = reference_hists[0].GetPaintedGraph().GetYaxis().GetTitle()
            index = ROOT.gROOT.GetListOfCanvases().IndexOf(ctmp)
            ROOT.gROOT.GetListOfCanvases().RemoveAt(index)
        canvas = PT.plot_obj(reference_hists[0], plot_config)
        for hist in reference_hists[1:]:
            PT.add_object_to_canvas(canvas, hist, plot_config)
        for hist in hists:
            hist.SetName(hist.GetName() + "_%i" % hists.index(hist))
            try:
                plot_config.color = self.color_palette[hists.index(hist)]
            except IndexError:
                _logger.warning("Run of colors in palette. Using black as default")
                plot_config.color = ROOT.kBlack
            PT.add_object_to_canvas(canvas, hist, plot_config)
        canvas.Modified()
        canvas.Update()
        FM.decorate_canvas(canvas, plot_config)
        if labels is None:
            labels = ["reference"] + [""] * len(hists)
        if hasattr(plot_config, "labels"):
            labels = plot_config.labels
        if hasattr(plot_config, "labels"):
            labels = plot_config.labels
        if len(labels) != len(hists) + len(reference_hists):
            _logger.error("Not enough labels provided. Received %i labels for %i histograms" % (len(labels),
                                                                                                len(hists) + 1))
            labels += [""] * (len(hists) - len(labels))
        FM.add_legend_to_canvas(canvas, labels=labels, **plot_config.legend_options)
        if plot_config.stat_box:
            FM.add_stat_box_to_canvas(canvas)
        if hasattr(plot_config, "ratio_config"):
            plot_config = plot_config.ratio_config
        plot_config.name = "ratio_" + plot_config.name
        canvas_ratio = RatioPlotter(reference=reference_hists[0], compare=reference_hists[1:] + hists,
                                    plot_config=plot_config).make_ratio_plot()
        canvas_combined = PT.add_ratio_to_canvas(canvas, canvas_ratio)
        self.output_handle.register_object(canvas)
        self.output_handle.register_object(canvas_combined)
