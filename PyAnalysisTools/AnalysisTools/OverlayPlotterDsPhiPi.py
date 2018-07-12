import ROOT
import math
from ROOT import gROOT
from PyAnalysisTools.base.YAMLHandle import YAMLLoader
from PyAnalysisTools.base.OutputHandle import OutputFileHandle
from PyAnalysisTools.ROOTUtils.ObjectHandle import *
from PyAnalysisTools.PlottingUtils.Formatting import add_atlas_label

class OverlayPlotterDsPhiPi(object):
   def __init__(self, **kwargs):
       self.input_dict = {}
       self.file_dict = {}
       self.variable_config = None
       self.parameter_list = []
       self.output_dir = kwargs["output_dir"]
       if kwargs["variable_config_file"]:
           self.variable_config = YAMLLoader.read_yaml(kwargs["variable_config_file"])
           self.variable_config.pop("common")
       if kwargs["input_config"]:
           self.input_dict = YAMLLoader.read_yaml(kwargs["input_config"])
           common_config = self.input_dict.pop("common")
           self.xtitle = common_config.get("xtitle", "")
           self.ytitle = common_config.get("ytitle", "")
           self.parameter_list = common_config.get("parameters",[])
           self.logyrange = common_config.get("logyrange",[1.,2.])
           for key in self.input_dict.keys():
               self.file_dict[key] =  ROOT.TFile(self.input_dict[key]["Path"])
       self.output_handle = OutputFileHandle(output_dir=self.output_dir)
       self.output_handle.set_n_plots_per_page(1)

   def get_hist_from_file(self, file, canvas_name, hist_name, legend):
       canvas = file.Get(canvas_name)
       hist = get_objects_from_canvas_by_name(canvas, hist_name)[0]
       if hist:
          hist.SetTitle(legend)
          hist.SetName((hist.GetName()+legend).replace(" ",""))
          return hist
       else:
          return None

   def get_hist_list_color_list(self, variable, parameter_name=""):
       hist_list , color_list = [], []
       hist_name = ""
       for key in sorted(self.input_dict.keys()):
           file = self.file_dict[key]
           if variable and parameter_name:
              canvas_name = variable
              hist_name = variable+"_"+parameter_name
           elif variable:
              canvas_name = variable
              hist_name = variable + self.input_dict[key].get("Label","")
           else:
              canvas_name = self.input_dict[key]["CanvasName"]
              hist_name = self.input_dict[key]["HistName"]
           legend = self.input_dict[key]["Legend"]
           hist = self.get_hist_from_file(file, canvas_name, hist_name, legend)
           if hist:
              hist_list.append(hist)
              color_list.append(self.input_dict[key].get("Color", 1))
       return hist_list, color_list

   def format_and_draw_hist(self, hist_list, color_list, variable, parameter=""):
       canvas = ROOT.TCanvas(variable, variable, 800, 600)
       hist_base = None
       self.legend = ROOT.TLegend(0.5,0.71,0.91,0.93)
       max = 0
       min = 999999
       # Set base hist to scale to
       for hist in hist_list:
           hist.ResetStats()
           if "Data16 Ds" in hist.GetTitle() and "N_" in self.ytitle:
              hist_base = hist
       # Formatting
       for hist in hist_list:
           if hist.GetBinContent(1) < 0:
              hist.Scale(-1.)
           if hist_base:
              hist.Scale(hist_base.Integral()/hist.Integral())
           hist.SetMarkerColor(color_list.pop(0))
           hist.SetLineColor(hist.GetMarkerColor())
           self.legend.AddEntry(hist, hist.GetTitle())
           hist.SetYTitle(self.ytitle)
           hist.SetYTitle(parameter)
           hist.SetXTitle(variable)
           if hist.GetMaximum() > max:
              max = hist.GetMaximum()
           if hist.GetMinimum() < min:
              min = hist.GetMinimum()
       # Draw
       for hist in hist_list:
           if self.logyrange:
              ROOT.gPad.SetLogy()
              hist.SetAxisRange(self.logyrange[0],self.logyrange[1], "Y")
              hist.SetAxisRange(-1, 10, "X")
           elif max > 1.2*min:
              hist.SetAxisRange(0., 1.4*max, "Y")
           else:
              hist.SetAxisRange(min-2*(max-min), max+2*(max-min), "Y")
           hist.Draw("same P")
           hist.Draw("same L1hist")
       if hist_base:
          self.legend.AddEntry("Ds", "N_{Ds}: " + str(int(round(hist_base.Integral()))))
       if parameter:
          canvas.Update()
          const = self.parameter_list[parameter]
          const_line = ROOT.TLine()
          const_line.SetLineWidth(14)
          const_line.SetLineColor(1)
          const_line.DrawLine(ROOT.gPad.GetUxmin(), const, ROOT.gPad.GetUxmax(), const)
          self.legend.AddEntry(const_line, "Data16 full stat fit")
       self.legend.Draw("same")
       add_atlas_label(canvas, "Internal", pos={'x': 0.2, 'y': 0.87})
       return canvas

   def make_parameter_per_slice_plot_book(self, variable_keys, parameter):
       for variable in sorted(variable_keys):
           hist_list, color_list = self.get_hist_list_color_list(variable, parameter)
           canvas = self.format_and_draw_hist(hist_list,color_list, variable, parameter)
           canvas.SetName(variable+"_"+parameter)
           self.legend.Draw("same")
           self.output_handle.register_object(canvas)
       self.output_handle.set_plot_book_name(parameter)
       self.output_handle.make_plot_book()
       self.output_handle.clear_objects()

   def make_plots(self):
       gROOT.SetBatch(True)
       if self.variable_config and self.parameter_list:
            for parameter in self.parameter_list.keys():
                self.make_parameter_per_slice_plot_book(self.variable_config.keys(), parameter)
       elif self.variable_config and not self.parameter_list:
            self.make_parameter_per_slice_plot_book(self.variable_config.keys(), "")
       else:
            hist_list, color_list = self.get_hist_list_color_list(None)
            canvas = self.format_and_draw_hist(hist_list,color_list, self.xtitle)
            self.output_handle.dump_canvas(canvas)
