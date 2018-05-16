import ROOT
import pathos.multiprocessing as mp
from PyAnalysisTools.base.YAMLHandle import YAMLLoader
from PyAnalysisTools.AnalysisTools.Fitter import Fitter
from PyAnalysisTools.AnalysisTools.FitHelpers import get_Ds_count, get_D_count, get_background_count, get_Ds_width, get_Ds_mass
from PyAnalysisTools.base.OutputHandle import OutputFileHandle
from PyAnalysisTools.PlottingUtils.Formatting import add_text_to_canvas
from PyAnalysisTools.ROOTUtils.ObjectHandle import *

class FitterPerSlice(object):

   def get_list_of_variables(self, variable_dict):
       list_of_variables = []
       variable_dict.pop("common")
       for key in variable_dict:
           variable_dict[key]["key"]=key
           list_of_variables.append(variable_dict[key])
       return list_of_variables
    
   def get_list_of_slices(self, slicing_variable):
       variable_name = slicing_variable["dist"]
       n_bin = slicing_variable["bins"]
       variable_min = slicing_variable["xmin"]
       variable_max = slicing_variable["xmax"]
       interval = (variable_max - variable_min)/n_bin
       list_of_slices = []
       bin_center = 0
       for i in range(0, n_bin):
           bin_lower = "%.6g"%(variable_min + i*interval)
           bin_upper = "%.6g"%(variable_min + (i+1)*interval)
           bin_center = variable_min + (i+0.5)*interval
           if i is 0:
              list_of_slices.append([variable_name+ "<" + bin_upper, bin_center])
           elif i is n_bin-1:
              list_of_slices.append([bin_lower + "<=" + variable_name, bin_center])
           else:
              list_of_slices.append([bin_lower + "<=" + variable_name + "&&" + variable_name+ "<" + bin_upper, bin_center])
       return list_of_slices

   def format_and_draw_hists(self, canvas, hist_list, slicing_variable):
       canvas.cd()
       for hist in hist_list:
           hist.ResetStats()
           hist.GetXaxis().SetTitle(slicing_variable["xtitle"])
           hist.Draw("histsame")
       return canvas

   def __init__(self, **kwargs):
       self.fitter = Fitter(**kwargs)
       if "ncpu" in kwargs:
           self.ncpu = kwargs["ncpu"]
       if "slicing_variable_config_file" in kwargs:
           slicing_variables_config = YAMLLoader.read_yaml(kwargs["slicing_variable_config_file"])
           self.slicing_variables_config = self.get_list_of_variables(slicing_variables_config)
       self.output_handle = OutputFileHandle(output_dir=kwargs["output_dir"])

   def fit_per_slice(self, variable_config):
       list_of_slices_and_bin_center = self.get_list_of_slices(variable_config)
       list_n_Ds, list_n_Ds_error = [], []
       list_n_Bkg, list_n_Bkg_error = [], []
       list_w_Ds, list_w_Ds_error = [], []
       list_m_Ds, list_m_Ds_error = [], []
       for [individual_slice, bin_center] in list_of_slices_and_bin_center:
           #define fit name and fit
           fit_name = variable_config["dist"]+"_"+"%.3g"%(bin_center)
           model, fit_result, canvas_slice = self.fitter.fit(False, [individual_slice])
           #get number of D, Ds, background events
           n_D, n_D_error = get_D_count(model)
           n_Ds, n_Ds_error = get_Ds_count(model)
           n_Bkg, n_Bkg_error = get_background_count(model)
           w_Ds, w_Ds_error = get_Ds_width(model)
           m_Ds, m_Ds_error = get_Ds_mass(model)
           #add fit name and register the canvas
           add_text_to_canvas(canvas_slice, fit_name, pos={"x": 0.16, "y": 0.96})
           canvas_slice.SetName(fit_name)
           self.output_handle.register_object(canvas_slice)
           #append Ds count, Bkg count, canvas
           list_n_Ds.append(n_Ds)
           list_n_Ds_error.append(n_Ds_error)
           list_n_Bkg.append(n_Bkg)
           list_n_Bkg_error.append(n_Bkg_error)
           list_w_Ds.append(w_Ds)
           list_w_Ds_error.append(w_Ds_error)
           list_m_Ds.append(m_Ds)
           list_m_Ds_error.append(m_Ds_error)
       #make plot book of individual hists
       self.output_handle.set_n_plots_per_page(30)
       self.output_handle.set_plot_book_name("individual_fit_" + variable_config["dist"])
       self.output_handle.make_plot_book()
       #book histogram
       hist_Ds_count = ROOT.TH1F(variable_config["key"]+"_DsCount", "",
                                 variable_config["bins"], variable_config["xmin"],
                                 variable_config["xmax"])
       hist_Bkg_count = ROOT.TH1F(variable_config["key"]+"_BkgCount", "",
                                  variable_config["bins"], variable_config["xmin"],
                                  variable_config["xmax"])
       hist_Ds_width = ROOT.TH1F(variable_config["key"]+"_DsWidth", "",
                                 variable_config["bins"], variable_config["xmin"],
                                 variable_config["xmax"])
       hist_Ds_mass = ROOT.TH1F(variable_config["key"]+"_DsMass", "",
                                variable_config["bins"], variable_config["xmin"],
                                variable_config["xmax"])
       ROOT.SetOwnership(hist_Ds_count, False)
       ROOT.SetOwnership(hist_Bkg_count, False)
       ROOT.SetOwnership(hist_Ds_width, False)
       ROOT.SetOwnership(hist_Ds_mass, False)
       #fill histogram
       for i in range(0, variable_config["bins"]):
           hist_Ds_count.SetBinContent(i+1,list_n_Ds[i])
           hist_Ds_count.SetBinError(i+1, list_n_Ds_error[i])
           hist_Ds_width.SetBinContent(i+1,list_w_Ds[i])
           hist_Ds_width.SetBinError(i+1, list_w_Ds_error[i])
           hist_Ds_mass.SetBinContent(i+1,list_m_Ds[i])
           hist_Ds_mass.SetBinError(i+1, list_m_Ds_error[i])
           hist_Bkg_count.SetBinContent(i+1,list_n_Bkg[i])
           hist_Bkg_count.SetBinError(i+1, list_n_Bkg_error[i])
       #format and draw
       canvas = ROOT.TCanvas(variable_config["key"], "", 800, 600)
       ROOT.SetOwnership(canvas, False)
       self.format_and_draw_hists(canvas, [hist_Ds_count, hist_Bkg_count, hist_Ds_width, hist_Ds_mass], variable_config)
       return canvas

   def fit_all_slices(self):
       pool = mp.ProcessPool(self.ncpu)
       list_canvas = pool.map(self.fit_per_slice, self.slicing_variables_config)
       for canvas in list_canvas:
           self.output_handle.register_object(canvas)
       self.output_handle.write_and_close()
