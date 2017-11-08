import os
import re
import time
import ROOT
from PyAnalysisTools.base import _logger, InvalidInputError
from PyAnalysisTools.base import ShellUtils
from PyAnalysisTools.PlottingUtils.PlottingTools import retrieve_new_canvas


class SysOutputHandle(object):
    def __init__(self, **kwargs):
        if "output_dir" not in kwargs:
            _logger.warning("No output directory provied")
            kwargs.setdefault("output_dir", "./")
        kwargs.setdefault("sub_dir_name", "output")
        self.base_output_dir = kwargs["output_dir"]
        self.output_dir = resolve_output_dir(**kwargs)
        ShellUtils.make_dirs(self.output_dir)

    @staticmethod
    def resolve_output_dir(**kwargs):
        time_stamp = time.strftime("%Y%m%d_%H-%M-%S")
        output_dir = kwargs["output_dir"]
        if output_dir is None:
            output_dir = "./"
        if os.path.islink(output_dir):
            output_dir = os.readlink(output_dir)
        if re.search(r"([0-9]{8}_[0-9]{2}-[0-9]{2}-[0-9]{2})$", output_dir):
            return output_dir
        return os.path.join(output_dir, "{}_{}".format(kwargs["sub_dir_name"], time_stamp))

    def _set_latest_link(self, link):
        if os.path.exists(link):
            os.unlink(link)
        os.symlink(self.output_dir, link)

    def set_latest_link(self, overload=None):
        latest_link_path = os.path.join(self.base_output_dir, "latest")
        self._set_latest_link(latest_link_path)
        if overload:
            latest_link_path_overload = os.path.join(self.base_output_dir, "latest_%s" % overload)
            self._set_latest_link(latest_link_path_overload)


class OutputHandle(SysOutputHandle):
    def __init__(self, **kwargs):
        super(self.__class__, self).__init__(**kwargs)
        self.file_list = []

    def register_file_list(self, file_list):
        for file_name in file_list:
            self.register_file(file_name)

    def register_file(self, file_name):
        self.file_list.append(file_name)

    def move_all_files(self, overload=None):
        for abs_file_name in self.file_list:
            try:
                _, file_name = os.path.split(abs_file_name)
                ShellUtils.move(abs_file_name, os.path.join(self.output_dir, file_name))
            except IOError:
                _logger.error("Unable to move %s to %s" % (abs_file_name, self.output_dir))
                raise
        self.set_latest_link(overload)


class OutputFileHandle(SysOutputHandle):
    def __init__(self, overload=None, **kwargs):
        super(self.__class__, self).__init__(**kwargs)
        self.objects = dict()
        self.attached = False
        self.overload = overload
        kwargs.setdefault("output_file", "output.root")
        self.output_file_name = kwargs["output_file"]
        self.output_file = None
        kwargs.setdefault("make_plotbook", False)
        self.enable_make_plot_book = kwargs["make_plotbook"]

    def attach_file(self):
        if not self.attached:
            self.output_file = ROOT.TFile.Open(os.path.join(self.output_dir, self.output_file_name), "RECREATE")
            self.output_file.cd()
            self.attached = True

    def dump_canvas(self, canvas, name=None, tdir=None):
        #re-draw canvas to update internal reference in gPad
        canvas.Draw()
        ROOT.gPad.Update()
        extension = ".pdf"
        if not name:
            name = canvas.GetName()
        output_path = self.output_dir
        if tdir is not None:
            output_path = os.path.join(output_path, tdir)
            ShellUtils.make_dirs(output_path)
        canvas.SaveAs(os.path.join(output_path, name + extension))

    #todo: quite fragile as assumptions on bucket size are explicitly taken
    def _make_plot_book(self, bucket, counter, prefix="plot_book"):
        plot_book_canvas = retrieve_new_canvas("{:s}_{:d}".format(prefix, counter), "", 1500, 2000)
        plot_book_canvas.Divide(3, 4)
        for i in range(len(bucket)):
            plot_book_canvas.cd(i+1)
            bucket[i].DrawClonePad()
        self.dump_canvas(plot_book_canvas)

    def make_plot_book(self):
        all_canvases = filter(lambda obj: isinstance(obj, ROOT.TCanvas), self.objects.values())
        ratio_plots = filter(lambda c: "ratio" in c.GetName(), all_canvases)
        plots = list(set(all_canvases) - set(ratio_plots))
        plots.sort(key=lambda i: i.GetName())
        ratio_plots.sort(key=lambda i: i.GetName())
        plots = [plots[i:i+9] for i in range(0, len(plots), 12)]
        ratio_plots = [ratio_plots[i:i + 9] for i in range(0, len(ratio_plots), 9)]
        for plot_bucket in plots:
            self._make_plot_book(plot_bucket, plots.index(plot_bucket))
        for plot_bucket in ratio_plots:
            self._make_plot_book(plot_bucket, ratio_plots.index(plot_bucket), prefix="plot_book_ratio")

    def write_to_file(self, obj, tdir=None):
        if tdir is not None:
            self.output_file.mkdir(tdir)
        else:
            tdir = "/"
        directory = self.output_file.GetDirectory(tdir)
        directory.cd()
        obj.Write()
        self.output_file.cd("/")

    def write_and_close(self):
        if self.enable_make_plot_book:
            self.make_plot_book()
        self.attach_file()
        for tdir, obj in self.objects.iteritems():
            if isinstance(obj, ROOT.TCanvas):
                self.dump_canvas(obj, tdir = tdir[0])
            self.write_to_file(obj, tdir[0])
            #obj.Write()
        self.output_file.Write()
        self.output_file.Close()
        _logger.info("Written file %s" % self.output_file.GetName())
        #self.set_latest_link(self.overload)

    def register_object(self, obj, tdir=None):
        _logger.debug("Adding object %s" % obj.GetName())
        if tdir is not None and not tdir.endswith("/"):
            tdir += "/"
        if isinstance(obj, ROOT.TTree):
            self.objects[(tdir, obj.GetName())] = obj.CloneTree()
        else:
            self.objects[(tdir, obj.GetName())] = obj.Clone(obj.GetName() + "_clone")
