import os
import time
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

    def move_all_files(self):
        for abs_file_name in self.file_list:
            try:
                _, file_name = os.path.split(abs_file_name)
                ShellUtils.move(abs_file_name, os.path.join(self.output_dir, file_name))
            except IOError:
                _logger.error("Unable to move %s to %s" % (abs_file_name, self.output_dir))
                raise
        self.set_latest_link()

    def set_latest_link(self):
        latest_link_path = os.path.join(self.base_output_dir, "latest")
        if os.path.exists(latest_link_path):
            os.unlink(latest_link_path)
        os.symlink(self.output_dir, latest_link_path)