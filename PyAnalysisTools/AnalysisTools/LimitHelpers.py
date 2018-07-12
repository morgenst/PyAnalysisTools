import ROOT
import PyAnalysisTools.PlottingUtils.PlottingTools as pt
import PyAnalysisTools.PlottingUtils.Formatting as fm
from PyAnalysisTools.PlottingUtils.PlotConfig import PlotConfig


def read_workspace_from_file(file_name, ws_name = "w"):
    f = ROOT.TFile.Open(file_name)
    return f.Get(ws_name)


def get_fit_quality(file_name, ws_name, fr_name):
    ws = read_workspace_from_file(file_name, ws_name)
    fit_result = ws.obj(fr_name)
    return fit_result.status(), fit_result.covQual()


def make_cross_section_limit_plot(data, lumi=80., ytitle=None):
    data.sort()
    if ytitle is None:
        ytitle = "95% CL U.L on #sigma [pb]"
    pc = PlotConfig(name="xsec_limit", ytitle=ytitle, xtitle="m [GeV]", draw="ap", logy=True, lumi=lumi,
                    watermark="Internal")
    graph = ROOT.TGraph(len(data))
    for i, item in enumerate(data):
        print item
        graph.SetPoint(i, item[0], item[1] * item[2]/(lumi*1000.))
    graph.SetName("xsec_limit")
    canvas = pt.plot_obj(graph, pc)
    fm.decorate_canvas(canvas, pc)
    return canvas
