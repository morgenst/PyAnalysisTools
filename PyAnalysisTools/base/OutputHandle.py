import os
import time
import ROOT
from PyAnalysisTools.base import _logger, InvalidInputError
from PyAnalysisTools.base import ShellUtils


class SysOutputHandle(object):
    def __init__(self, **kwargs):
        if "output_dir" not in kwargs:
            _logger.error("No output directory provied")
            raise InvalidInputError("Missing output directory")
        kwargs.setdefault("sub_dir_name", "output")
        self.time_stamp = time.strftime("%Y%m%d_%H-%M-%S")
        self.base_output_dir = kwargs["output_dir"]
        self.output_dir = os.path.join(kwargs["output_dir"], "{}_{}".format(kwargs["sub_dir_name"], self.time_stamp))
        ShellUtils.make_dirs(self.output_dir)

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
        # todo: refactor using kwargs setdefault
        self.output_file_name = kwargs["output_file_name"]
        self.output_file = None

    def __del__(self):
        self._write_and_close()

    def attach_file(self):
        if not self.attached:
            self.output_file = ROOT.TFile.Open(os.path.join(self.output_dir, self.output_file_name), "RECREATE")
            self.output_file.cd()
            self.attached = True

    def dump_canvas(self, canvas, name=None):
        #re-draw canvas to update internal reference in gPad
        canvas.Draw()
        ROOT.gPad.Update()
        extension = ".pdf"
        if not name:
            name = canvas.GetName()
        canvas.SaveAs(os.path.join(os.path.join(self.output_dir,
                                                name + extension)))

    def _write_and_close(self):
        self.attach_file()
        for obj in self.objects.values():
            if isinstance(obj, ROOT.TCanvas):
                self.dump_canvas(obj)
            obj.Write()
        self.output_file.Write()
        self.output_file.Close()
        _logger.info("Written file %s" % self.output_file.GetName())
        #self.set_latest_link(self.overload)

    def register_object(self, obj, tdir=""):
        _logger.debug("Adding object %s" % obj.GetName())
        self.objects[tdir + obj.GetName()] = obj.Clone(obj.GetName() + "_clone")
