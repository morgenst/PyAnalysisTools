import ROOT
import PyAnalysisTools.PlottingUtils.PlottingTools as PT
import PyAnalysisTools.PlottingUtils.Formatting as FM
from copy import copy
from PyAnalysisTools.base import _logger, InvalidInputError
from PyAnalysisTools.ROOTUtils.FileHandle import FileHandle
from PyAnalysisTools.base.OutputHandle import OutputFileHandle
from PyAnalysisTools.PlottingUtils.PlotConfig import parse_and_build_plot_config
from PyAnalysisTools.ROOTUtils.ObjectHandle import get_objects_from_canvas_by_name, get_objects_from_canvas_by_type


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

    def parse_config(self):
        self.plot_configs, self.common_config = parse_and_build_plot_config(self.config_file)

    def get_instance(self, plot_config):
        if hasattr(plot_config, "dist") and hasattr(plot_config, "dist_ref") and hasattr(plot_config, "processes"):
            _logger.debug("Using SingleFileMultiReader instance")
            return SingleFileMultiDistReader(self.input_files, plot_config)
        if hasattr(plot_config, "dist") and not hasattr(plot_config, "dist_ref") and self.reference_file:
            _logger.debug("Using SingleFileMultiReader instance")
            return MultiFileSingleDistReader(self.reference_file, self.input_files, plot_config)

    def get_data(self):
        data = {}
        for plot_config in self.plot_configs:
            getter = self.get_instance(plot_config)
            data[plot_config] = getter.get_data()
        return data


class SingleFileMultiDistReader(ComparisonReader):
    def __init__(self, input_files, plot_config):
        self.file_handle = FileHandle(file_name=input_files)
        self.plot_config = plot_config

    def get_data(self):
        reference_canvas = self.file_handle.get_object_by_name(self.plot_config.dist_ref)
        compare_canvas = self.file_handle.get_object_by_name(self.plot_config.dist)
        reference = get_objects_from_canvas_by_name(reference_canvas, self.plot_config.processes[0])[0]
        compare = get_objects_from_canvas_by_name(compare_canvas, self.plot_config.processes[0])
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
    def __init__(self, reference_file, input_files, plot_config):
        self.reference_file_handle = FileHandle(file_name=reference_file)
        self.file_handles = [FileHandle(file_name=fn) for fn in input_files]
        self.plot_config = plot_config

    def get_data(self):
        reference_canvas = self.reference_file_handle.get_object_by_name(self.plot_config.dist)
        #todo: generalise to arbitrary number of compare inputs
        #todo: generalise to type given by plot config
        compare_canvas = self.file_handles[0].get_object_by_name(self.plot_config.dist)
        reference = get_objects_from_canvas_by_type(reference_canvas, "TEfficiency")[0]
        compare = get_objects_from_canvas_by_type(compare_canvas, "TEfficiency")
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
        kwargs.setdefault("output_dir", "./")
        self.input_files = kwargs["input_files"]
        self.config_file = kwargs["config_file"]
        self.output_handle = OutputFileHandle(overload="comparison", output_file_name="Compare.root", **kwargs)
        self.color_palette = [ROOT.kRed, ROOT.kBlue, ROOT.kGreen, ROOT.kCyan]
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

    @staticmethod
    def calculate_efficiency_ratio(hist, reference):
        ratio_graph = reference.GetPaintedGraph().Clone("ratio_"+hist.GetName())
        nbins = ratio_graph.GetN()
        for b in range(nbins):
            eff_compare = hist.GetEfficiency(b)
            eff_ratio = hist.GetEfficiency(b)
            if eff_ratio == 0.:
                ratio = 0.
            else:
                ratio = eff_compare/eff_ratio
            x = ROOT.Double(0.)
            y = ROOT.Double(0.)
            ratio_graph.GetPoint(b, x, y)
            ratio_graph.SetPoint(b, x, ratio)
        return ratio_graph

    @staticmethod
    def calculate_ratio(hist, reference):
        if isinstance(reference, ROOT.TEfficiency):
            return ComparisonPlotter.calculate_efficiency_ratio(hist, reference)
        ratio_hist = hist.Clone("ratio_" + hist.GetName())
        ratio_hist.Divide(reference)
        FM.set_title_y(ratio_hist, "ratio")
        return ratio_hist

    def calculate_ratios(self, hists, reference):
        ratio_plot_config = self.common_config.ratio_config
        if not isinstance(reference, ROOT.TEfficiency):
            ratio_plot_config.xtitle = reference.GetXaxis().GetTitle()
        else:
            ratio_plot_config.xtitle = reference.GetPaintedGraph().GetXaxis().GetTitle()
        ratios = [self.calculate_ratio(hist, reference) for hist in hists]
        if not isinstance(reference, ROOT.TEfficiency):
            canvas = PT.plot_histograms(ratios, ratio_plot_config)
        else:
            canvas = PT.plot_graph(ratios[0], ratio_plot_config)
        return canvas

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
            y_max = 1.1 * max([item.GetMaximum() for item in hists] + [reference_hist.GetMaximum()])
        canvas = PT.plot_obj(reference_hist, plot_config, y_max=y_max)
        for hist in hists:
            hist.SetName(hist.GetName() + "_%i" % hists.index(hist))
            plot_config.color = self.color_palette[hists.index(hist)]
            PT.add_histogram_to_canvas(canvas, hist, plot_config)
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
        FM.add_legend_to_canvas(canvas, labels=labels, xl=0.3, xh=0.5)
        if self.common_config.stat_box:
            FM.add_stat_box_to_canvas(canvas)
        canvas_ratio = self.calculate_ratios(hists, reference_hist)
        canvas_combined = PT.add_ratio_to_canvas(canvas, canvas_ratio)
        self.output_handle.register_object(canvas)
        self.output_handle.register_object(canvas_combined)

    def compare_objects(self):
        print "obsolete"
        pass
        self.parse_config()
        if hasattr(self.common_config, "colors"):
            self.update_color_palette()
        for plot_config in self.plot_configs:
            self.make_comparison_plot(plot_config)