import ROOT
import pathos.multiprocessing as mp
from PyAnalysisTools.base.YAMLHandle import YAMLLoader as YL
from PyAnalysisTools.AnalysisTools.Fitter import Fitter
from PyAnalysisTools.base.OutputHandle import OutputFileHandle
from PyAnalysisTools.PlottingUtils.Formatting import add_text_to_canvas
from PyAnalysisTools.ROOTUtils.ObjectHandle import *

class FitterPerSlice(object):

   def get_list_variables(self, variable_dict):
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

   def format_and_draw_hists(self, hist_list, variable_config):
       canvas = ROOT.TCanvas(variable_config["key"], "", 800, 600)
       ROOT.SetOwnership(canvas, False)
       for hist in hist_list:
           hist.ResetStats()
           hist.GetXaxis().SetTitle(variable_config["xtitle"])
           hist.Draw("histsame")
       return canvas

   def fill_parameter_collection(self, model, parameter_collection):
       it = model.getVariables().createIterator()
       for parameter in iter(it.Next, None):
           if parameter.GetName() not in parameter_collection.keys():
              parameter_collection[parameter.GetName()] = []
           val_and_error = {'val':parameter.getVal(), 'error':parameter.getError()}
           parameter_collection[parameter.GetName()].append(val_and_error)

   def make_parameter_plots(self, parameter_collection, variable_config):
       hists_parameters = []
       for key in parameter_collection:
           hist = ROOT.TH1F(variable_config["key"]+"_"+key, "",
                            variable_config["bins"], variable_config["xmin"],
                            variable_config["xmax"])
           ROOT.SetOwnership(hist, False)
           list_value_and_error = parameter_collection[key]
           for i in range(0, len(list_value_and_error)):
               hist.SetBinContent(i+1, list_value_and_error[i]["val"])
               hist.SetBinError(i+1, list_value_and_error[i]["error"])
           hists_parameters.append(hist)
       return hists_parameters

   def __init__(self, **kwargs):
       self.fitter = Fitter(**kwargs)
       if "ncpu" in kwargs:
           self.ncpu = kwargs["ncpu"]
       if "slicing_variable_config_file" in kwargs:
           slicing_variables_config = YL.read_yaml(kwargs["slicing_variable_config_file"])
           self.slicing_variables_config = self.get_list_variables(slicing_variables_config)
       self.output_handle = OutputFileHandle(output_dir=kwargs["output_dir"])

   def fit_per_slice(self, variable_config):
       list_of_slices_and_bin_center = self.get_list_of_slices(variable_config)
       parameter_collection = {}
       for [individual_slice, bin_center] in list_of_slices_and_bin_center:
           #perform fit
           model, fit_result, canvas_slice = self.fitter.fit(False, [individual_slice])
           #add fit name to canvas
           fit_name = variable_config["dist"]+"_"+'{0:010.3f}'.format(bin_center)
           add_text_to_canvas(canvas_slice, fit_name, pos={"x": 0.16, "y": 0.96})
           canvas_slice.SetName(fit_name)
           #append individual fits, individual parameters
           self.output_handle.register_object(canvas_slice)
           self.fill_parameter_collection(model, parameter_collection)
       #make plot book of individual fits
       self.output_handle.set_n_plots_per_page(len(list_of_slices_and_bin_center))
       self.output_handle.set_plot_book_name("individual_fit_" + variable_config["dist"])
       self.output_handle.make_plot_book()
       #make plots of individual parameters
       hists_parameters = self.make_parameter_plots(parameter_collection, variable_config)
       canvas = self.format_and_draw_hists(hists_parameters, variable_config)
       return canvas

   def fit_all_slices(self):
       pool = mp.ProcessPool(self.ncpu)
       list_canvas = pool.map(self.fit_per_slice, self.slicing_variables_config)
       for canvas in list_canvas:
           self.output_handle.register_object(canvas)
       self.output_handle.write_and_close()
       return self.output_handle.output_file.GetName()
