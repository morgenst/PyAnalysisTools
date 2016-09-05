import os
import time
import ROOT
from PyAnalysisTools.base import _logger
from PyAnalysisTools.base import ShellUtils


class OutputHandle(object):
    def __init__(self, output_dir, sub_dir_name="output"):
        self.time_stamp = time.strftime("%Y%m%d_%H-%M-%S")
        self.base_output_dir = output_dir
        self.output_dir = os.path.join(output_dir, "{}_{}".format(sub_dir_name, self.time_stamp))
        ShellUtils.make_dirs(self.output_dir)
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


class OutputFileHandle(object):
    def __init__(self, output_file_name, output_path=None):
        if output_path:
            output_file_name = os.path.join(output_path, output_file_name)
        self.output_file_name = output_file_name
        self.output_path = output_path
        if output_path is None:
            self.output_path = "./"
        self.objects = dict()
        self.attached = False

    def __del__(self):
        self._write_and_close()

    def attach_file(self):
        if not self.attached:
            self.output_file = ROOT.TFile.Open(self.output_file_name, "RECREATE")
            self.output_file.cd()
            self.attached = True

    def dump_canvas(self, canvas, name=None):
        extension = ".pdf"
        if not name:
            name = canvas.GetName()
        canvas.SaveAs(os.path.join(os.path.join(self.output_path,
                                                name + extension)))

    def _write_and_close(self):
        self.attach_file()
        for obj in self.objects.values():
            obj.Write()
            if isinstance(obj, ROOT.TCanvas):
                self.dump_canvas(obj)
        self.output_file.Write()
        self.output_file.Close()
        _logger.info("Written file %s" % self.output_file.GetName())

    def register_object(self, obj, tdir=""):
        self.objects[tdir + obj.GetName()] = obj
