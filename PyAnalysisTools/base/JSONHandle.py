from builtins import object
import json
import os
from PyAnalysisTools.base import _logger
from PyAnalysisTools.base.Singleton import Singleton
from PyAnalysisTools.base.ShellUtils import copy
from future.utils import with_metaclass


class JSONHandle(with_metaclass(Singleton, object)):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('copy', False)
        self.data = {}
        self.file_name = os.path.join(args[0], "config.json")
        self.copy = kwargs['copy']
        self.input_file = None

    def add_args(self, **kwargs):
        for val, arg in list(kwargs.items()):
            self.data[val] = arg

    def reset_path(self, path):
        self.file_name = os.path.join(path, "config.json")

    def dump(self):
        if self.copy:
            if self.input_file is None:
                _logger.warning('Try copying json file, but no input file provided')
                return
            copy(self.input_file, self.file_name)
            return
        with open(self.file_name, 'w') as outfile:
            json.dump(self.data, outfile)

    def load(self):
        self.copy = True
        self.input_file = self.file_name
        with open(self.file_name, 'r') as input_file:
            return json.load(input_file)
