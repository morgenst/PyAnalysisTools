import ROOT
import PyAnalysisTools.PlottingUtils.PlottingTools as PT
import PyAnalysisTools.PlottingUtils.Formatting as FM
from copy import copy
from PyAnalysisTools.PlottingUtils import set_batch_mode
from PyAnalysisTools.base import _logger, InvalidInputError
from PyAnalysisTools.ROOTUtils.FileHandle import FileHandle
from PyAnalysisTools.base.OutputHandle import OutputFileHandle
from PyAnalysisTools.PlottingUtils.PlotConfig import parse_and_build_plot_config, get_histogram_definition, \
    expand_plot_config
from PyAnalysisTools.ROOTUtils.ObjectHandle import get_objects_from_canvas_by_name, get_objects_from_canvas_by_type, get_objects_from_canvas
from PyAnalysisTools.PlottingUtils.RatioPlotter import RatioPlotter


class ComparisonReader(object):
    def __init__(self, **kwargs):
        if not "config_file" in kwargs:
            _logger.error("No config file provided")
            raise InvalidInputError("Missing config")
        if not "input_files" in kwargs:
            _logger.error("No input file provided")
            raise InvalidInputError("Missing input files")
        kwargs.setdefault("reference_file", None)
        self.plot_configs = kwargs["plot_configs"]
        self.common_config = kwargs["common_config"]
        self.input_files = kwargs["input_files"]
        self.reference_file = kwargs["reference_file"]
        self.tree_name = kwargs["tree_name"]
        for opt, val in kwargs.iteritems():
            if not hasattr(self, opt):
                setattr(self, opt, val)

    def parse_config(self):
        self.plot_configs, self.common_config = parse_and_build_plot_config(self.config_file)

    def get_instance(self, plot_config):
        if hasattr(plot_config, "dist") and hasattr(plot_config, "dist_ref"):# and hasattr(plot_config, "processes"):
            _logger.debug("Using SingleFileMultiReader instance")
            return SingleFileMultiDistReader(input_files=self.input_files, plot_config=plot_config, tree_name=self.tree_name)
        if hasattr(plot_config, "dist") and not hasattr(plot_config, "dist_ref") and self.reference_file:
            _logger.debug("Using MultiFileSingleDistReader instance")
            return MultiFileSingleDistReader(plot_config=plot_config, **self.__dict__)

    def get_data(self):
        data = {}
        for plot_config in self.plot_configs:
            getter = self.get_instance(plot_config)
            data[plot_config] = getter.get_data()
        return data

    def make_plot(self, file_handle, plot_config, tree_name=None, use_plot_config_name=False):
        hist = get_histogram_definition(plot_config)
        hist.SetName(hist.GetName() + file_handle.process)
        if tree_name is None:
            tree_name = self.tree_name
        try:
            file_handle.fetch_and_link_hist_to_tree(tree_name, hist, plot_config.dist, None,
                                                    tdirectory="Nominal")
            hist.SetName(hist.GetName() + "_" + file_handle.process)
            _logger.debug("try to access config for process %s" % file_handle.process)
        except Exception as e:
            raise e
        return hist


class SingleFileMultiDistReader(ComparisonReader):
    def __init__(self, **kwargs):
        input_files = kwargs["input_files"]
        plot_config = kwargs["plot_config"]
        if isinstance(input_files, list):
            if len(input_files) > 1:
                _logger.error("Privided {:d} input files for single file reader. "
                              "Using just first".format(len(input_files)))
            input_files = input_files[0]
        self.file_handle = FileHandle(file_name=input_files)
        self.plot_config = plot_config
        self.tree_name = kwargs["tree_name"]

    def get_data(self):
        try:
            reference_canvas = self.file_handle.get_object_by_name(self.plot_config.dist_ref, tdirectory="Nominal")
            compare_canvas = self.file_handle.get_object_by_name(self.plot_config.dist, tdirectory="Nominal")
            if hasattr(self.plot_config, "retrieve_by") and self.plot_config.retrieve_by == "type":
                reference = get_objects_from_canvas_by_type(reference_canvas, "TH1F")[0]
                compare = get_objects_from_canvas_by_type(compare_canvas, "TH1F")
            else:
                reference = get_objects_from_canvas_by_name(reference_canvas, self.plot_config.processes[0])[0]
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


class MultiFileSingleDistReader(ComparisonReader):
    """
    Read same distribitionS from reference and input files
    """
    def __init__(self, **kwargs):
        reference_file = kwargs["reference_file"]
        input_files = kwargs["input_files"]
        plot_config = kwargs["plot_config"]
        self.reference_file_handle = FileHandle(file_name=reference_file, switch_off_process_name_analysis=True)
        self.file_handles = [FileHandle(file_name=fn, switch_off_process_name_analysis=True) for fn in input_files]
        self.plot_config = plot_config
        self.tree_name = kwargs["tree_name"]
        self.reference_tree_name = copy(self.tree_name)
        if "reference_tree_name" in kwargs and not kwargs["reference_tree_name"] is None:
            self.reference_tree_name = kwargs["reference_tree_name"]

    def get_data(self):
        try:
            reference_canvas = self.reference_file_handle.get_object_by_name(self.plot_config.dist)
            #todo: generalise to arbitrary number of compare inputs
            #todo: generalise to type given by plot config
            compare_canvas = self.file_handles[0].get_object_by_name(self.plot_config.dist)
            if hasattr(self.plot_config, "retrieve_by") and "type" in self.plot_config.retrieve_by:
                obj_type = self.plot_config.retrieve_by.replace("type:", "")
            reference = get_objects_from_canvas_by_type(reference_canvas, obj_type)[0]
            compare = get_objects_from_canvas_by_type(compare_canvas, obj_type)
        except ValueError:
            plot_config_ref = copy(self.plot_config)
            plot_config_ref.name += "_reference"
            reference = self.make_plot(self.reference_file_handle, plot_config_ref, self.reference_tree_name)
            compare = [self.make_plot(file_handle, self.plot_config) for file_handle in self.file_handles]
            reference.SetDirectory(0)
            map(lambda obj: obj.SetDirectory(0), compare)
        return reference, compare


class ComparisonPlotter(object):
    def __init__(self, **kwargs):
        if not "input_files" in kwargs:
            _logger.error("No input files provided")
            raise InvalidInputError("Missing input files")
        if not "config_file" in kwargs:
            _logger.error("No config file provided")
            raise InvalidInputError("Missing config")
        if not "output_dir" in kwargs:
            _logger.warning("No output directory given. Using ./")
        kwargs.setdefault("batch", True)
        kwargs.setdefault("tree_name", None)
        kwargs.setdefault("output_dir", "./")
        set_batch_mode(kwargs["batch"])
        self.input_files = kwargs["input_files"]
        self.config_file = kwargs["config_file"]
        self.output_handle = OutputFileHandle(overload="comparison", output_file_name="Compare.root", **kwargs)
        self.color_palette = [ROOT.kRed, ROOT.kBlue, ROOT.kGreen, ROOT.kCyan, ROOT.kPink, ROOT.kOrange, ROOT.kBlue-4,
                              ROOT.kRed+3, ROOT.kGreen-2]
        self.parse_config()
        for attr, value in kwargs.iteritems():
            if not hasattr(self, attr):
                setattr(self, attr, value)
        self.analyse_plot_config()
        self.getter = ComparisonReader(plot_configs=self.plot_configs, common_config=self.common_config, **kwargs)

    def parse_config(self):
        self.plot_configs, self.common_config = parse_and_build_plot_config(self.config_file)

    def analyse_plot_config(self):
        pc = next((pc for pc in self.plot_configs if pc.name == "parse_from_file"), None)
        if pc is None:
            return
        if not hasattr(self, "reference_file"):
            _logger.error("Request to parse plot configs from file, but no reference file given. Breaking up!")
            exit(0)
        file_handle = FileHandle(file_name=self.reference_file)
        objects = file_handle.get_objects_by_type("TCanvas")
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
        reference_hist = data[0]
        hists = data[1]
        y_max = None
        if not isinstance(reference_hist, ROOT.TEfficiency):
            yscale = 1.3
            ymax = yscale * max([item.GetMaximum() for item in hists] + [reference_hist.GetMaximum()])
            plot_config.yscale = yscale
            plot_config.ymax = ymax
        else:
            ctmp = ROOT.TCanvas("ctmp", "ctmp")
            ctmp.cd()
            reference_hist.Draw("ap")
            ROOT.gPad.Update()
            reference_hist.GetPaintedGraph().GetXaxis().GetTitle()
            plot_config.xtitle = reference_hist.GetPaintedGraph().GetXaxis().GetTitle()
            plot_config.ytitle = reference_hist.GetPaintedGraph().GetYaxis().GetTitle()
            index = ROOT.gROOT.GetListOfCanvases().IndexOf(ctmp)
            ROOT.gROOT.GetListOfCanvases().RemoveAt(index)
        canvas = PT.plot_obj(reference_hist, plot_config)

        for hist in hists:
            hist.SetName(hist.GetName() + "_%i" % hists.index(hist))
            plot_config.color = self.color_palette[hists.index(hist)]
            PT.add_object_to_canvas(canvas, hist, plot_config)
        canvas.Modified()
        canvas.Update()
        FM.decorate_canvas(canvas, self.common_config)
        labels = ["reference"] + [""] * len(hists)
        if hasattr(self.common_config, "labels") and not hasattr(plot_config, "labels"):
            labels = self.common_config.labels
        if hasattr(plot_config, "labels"):
            labels = plot_config.labels
        if len(labels) != len(hists) + 1:
            _logger.error("Not enough labels provided. Received %i labels for %i histograms" % (len(labels),
                                                                                               len(hists) + 1))
            labels += [""] * (len(hists) - len(labels))
        FM.add_legend_to_canvas(canvas, labels=labels, **plot_config.legend_options)
        if self.common_config.stat_box:
            FM.add_stat_box_to_canvas(canvas)
        if hasattr(self.common_config, "ratio_config"):
            plot_config = self.common_config.ratio_config
        plot_config.name = "ratio_" + plot_config.name
        canvas_ratio = RatioPlotter(reference=reference_hist, compare=hists, plot_config=plot_config).make_ratio_plot()
        canvas_combined = PT.add_ratio_to_canvas(canvas, canvas_ratio)
        self.output_handle.register_object(canvas)
        self.output_handle.register_object(canvas_combined)
