import json
import os
from PyAnalysisTools.base.Singleton import Singleton
from PyAnalysisTools.base.ShellUtils import copy


class JSONHandle(object):
    __metaclass__ = Singleton

    def __init__(self, *args, **kwargs):
        self.data = {}
        self.file_name = os.path.join(args[0], "config.json")
        self.copy = False
        self.input_file = None

    def add_args(self, **kwargs):
        for val, arg in kwargs.iteritems():
            self.data[val] = arg

    def reset_path(self, path):
        self.file_name = os.path.join(path, "config.json")

    def dump(self):
        if self.copy:
            copy(self.input_file, self.file_name)
            return
        with open(self.file_name, 'w') as outfile:
            json.dump(self.data, outfile)

    def load(self):
        self.copy = True
        self.input_file = self.file_name
        with open(self.file_name, 'r') as input_file:
            return json.load(input_file)
