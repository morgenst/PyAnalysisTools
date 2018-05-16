import ROOT
import math
from ROOT import gROOT
from PyAnalysisTools.base.YAMLHandle import YAMLLoader
from PyAnalysisTools.base.OutputHandle import OutputFileHandle
from PyAnalysisTools.ROOTUtils.ObjectHandle import *

class OverlayPlotterDsPhiPi(object):
   def __init__(self, **kwargs):
       self.input_dict = {}
       self.file_dict = {}
       self.xtitle, self.ytitle = "", ""
       self.variable_config = None
       if kwargs["variable_config_file"]:
           self.variable_config = YAMLLoader.read_yaml(kwargs["variable_config_file"])
           self.variable_config.pop("common")
       if kwargs["input_config"]:
           self.input_dict = YAMLLoader.read_yaml(kwargs["input_config"])
           common_config = self.input_dict.pop("common")
           self.ytitle = common_config.pop("ytitle")
           if "xtitle" in common_config.keys():
              self.xtitle = common_config.pop("xtitle")
           for key in self.input_dict.keys():
               self.file_dict[key] =  ROOT.TFile(self.input_dict[key]["Path"])
       self.output_handle = OutputFileHandle(output_dir=kwargs["output_dir"])
       self.output_handle.set_n_plots_per_page(20)

   def get_hist_from_file(self, file, canvas_name, hist_name, legend):
       canvas = ROOT.TCanvas("tmp"+canvas_name+legend, "", 800, 600)
       file.GetObject(canvas_name, canvas)
       hists = get_objects_from_canvas_by_name(canvas, hist_name)
       # Make sure exactly one hist is retrieved
       if hists and len(hists) is 1:
          hists[0].SetTitle(legend)
          return hists[0]
       else:
          return None

   def get_hist_list(self, variable):
       hist_list = []
       for key in self.input_dict.keys():
           file = self.file_dict[key]
           if variable:
              canvas_name = variable
              hist_name = variable+self.input_dict[key]["Label"]
           else:
              canvas_name = self.input_dict[key]["CanvasName"]
              hist_name = self.input_dict[key]["HistName"]
           legend = self.input_dict[key]["Legend"]
           hist = self.get_hist_from_file(file, canvas_name, hist_name, legend)
           if hist:
              hist_list.append(hist)
       return hist_list

   def format_and_draw_hist(self, hist_list, variable):
       canvas = ROOT.TCanvas(variable, variable, 800, 600)
       hist_base = None
       self.legend = ROOT.TLegend(0.58,0.78,0.84,0.93)
       color_list = [6, 35, 1, 2, 4, 3]
       max = 0
       min = 999999
       # Set base hist to scale to
       for hist in hist_list:
           hist.ResetStats()
           if "Data16 Ds" in hist.GetTitle() and "N_" in self.ytitle:
              hist_base = hist
       # Formatting
       for hist in hist_list:
           if hist_base:
              hist.Scale(hist_base.Integral()/hist.Integral())
           hist.SetMarkerColor(color_list.pop())
           hist.SetLineColor(hist.GetMarkerColor())
           self.legend.AddEntry(hist, hist.GetTitle())
           hist.SetYTitle(self.ytitle)
           hist.SetXTitle(variable)
           if hist.GetMaximum() > max:
              max = hist.GetMaximum()
           if hist.GetMinimum() < min:
              min = hist.GetMinimum()
       # Draw
       for hist in hist_list:
           if max > 1.2*min:
              hist.SetAxisRange(0., 1.4*max, "Y")
           else:
              hist.SetAxisRange(min-2*(max-min), max+2*(max-min), "Y")
           hist.Draw("same Lhist")
           hist.Draw("same E1P")
       if hist_base:
          self.legend.AddEntry("Ds", "N_{Ds}: " + str(int(round(hist_base.Integral()))))
       self.legend.Draw("same")
       return canvas

   def make_plots(self):
       gROOT.SetBatch(True)
       if self.variable_config:
          for variable in sorted(self.variable_config.keys()):
              hist_list = self.get_hist_list(variable)
              canvas = self.format_and_draw_hist(hist_list, variable)
              self.legend.Draw("same")
              self.output_handle.register_object(canvas)
          self.output_handle.make_plot_book()
       else:
          hist_list = self.get_hist_list(None)
          canvas = self.format_and_draw_hist(hist_list, self.xtitle)
          self.output_handle.dump_canvas(canvas)
