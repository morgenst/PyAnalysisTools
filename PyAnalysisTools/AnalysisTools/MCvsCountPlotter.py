import ROOT
from ROOT import gROOT
from PyAnalysisTools.base.YAMLHandle import YAMLLoader
from PyAnalysisTools.base.OutputHandle import OutputFileHandle
from PyAnalysisTools.ROOTUtils.ObjectHandle import *

class MCvsCountPlotter(object):
   def __init__(self, **kwargs):
       self.input_dict = {}
       self.file_dict = {}
       self.variable_list = []
       if "input_config" in kwargs:
           self.input_dict = YAMLLoader.read_yaml(kwargs["input_config"])
           self.ytitle = self.input_dict.pop("common").pop("ytitle")
           for key in self.input_dict.keys():
               self.file_dict[key] =  ROOT.TFile(self.input_dict[key]["Path"])
       if "variable_config_file" in kwargs:
           variable_config = YAMLLoader.read_yaml(kwargs["variable_config_file"])
           variable_config.pop("common")
           for key in variable_config:
               self.variable_list.append(key)
       self.output_handle = OutputFileHandle(output_dir=kwargs["output_dir"])

   def get_hist_from_file(self, variable):
       hist_list = []
       canvas = ROOT.TCanvas(variable+"_tmp", variable+"_tmp", 800, 600)
       for key in self.input_dict.keys():
           self.file_dict[key].GetObject(variable, canvas)
           hist = get_objects_from_canvas_by_name(canvas, variable+self.input_dict[key]["Label"])
           # Make sure exactly one hist is retrieved
           if hist and len(hist) is 1:
              hist[0].SetTitle(self.input_dict[key]["Legend"])
              hist_list += hist
       return hist_list

   def format_and_draw_hist(self, hist_list, variable):
       canvas = ROOT.TCanvas(variable, variable, 800, 600)
       hist_base = None
       self.legend = ROOT.TLegend(0.58,0.78,0.84,0.93)
       color_list = [1, 2, 4]
       max = 0
       min = 999999
       # Set base hist to scale to
       for hist in hist_list:
           hist.ResetStats()
           if "DsCount" in hist.GetTitle():
              hist_base = hist
       if not hist_base:
            hist_base = hist_list[0]
       # Formatting
       for hist in hist_list:
           if "DsCount" in hist_base.GetTitle():
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
              hist.SetAxisRange(0., 1.5*max, "Y")
           else:
              hist.SetAxisRange(min-2*(max-min), max+2*(max-min), "Y")
           hist.Draw("same Lhist")
           hist.Draw("same E1P")
       if "DsCount" in hist_base.GetTitle():
          self.legend.AddEntry("Ds", "N_{Ds}: " + str(int(round(hist_base.Integral()))))
       self.legend.Draw("same")
       return canvas

   def make_plots(self):
       gROOT.SetBatch(True)      
       for variable in self.variable_list:
           hist_list = self.get_hist_from_file(variable)
           canvas = self.format_and_draw_hist(hist_list, variable)
           self.output_handle.register_object(canvas)
       self.output_handle.write_and_close()
