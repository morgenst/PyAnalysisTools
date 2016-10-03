import os
import fnmatch
import glob
from PyAnalysisTools.base import _logger, InvalidInputError
from PyAnalysisTools.base.YAMLHandle import YAMLLoader


def flatten(dictionary, left_key="", separator="/"):
    flatten_list = []
    for key, value in dictionary.iteritems():
        if isinstance(value, dict):
            flatten_list += flatten(value, key)
        elif isinstance(value, list):
            for entry in value:
                flatten_list.append(separator.join([left_key, key, entry]))
        elif value is None:
            flatten_list.append(separator.join([left_key, key]))
    return flatten_list


def parse_dataset_list_from_file(file_name):
    return YAMLLoader.read_yaml(file_name)


def recursive_glob(path, pattern):
    matches = []
    for root, dir_names, file_names in os.walk(os.path.abspath(path)):
        for file_name in fnmatch.filter(file_names, pattern):
            matches.append(os.path.join(root, file_name))
    return matches


class Cleaner(object):
    def __init__(self, **kwargs):
        if "base_path" not in kwargs:
            _logger.error("No path provided")
            raise InvalidInputError("No path provided.")

        kwargs.setdefault("safe", True)
        self.base_path = os.path.abspath(kwargs["base_path"])
        self.safe = kwargs["safe"]
        self.keep_pattern = [".git", ".keep", ".svn"]
        self.deletion_list= []

    def retrieve_directory_list(self):
        directories = filter(lambda d: os.path.isdir(os.path.join(self.base_path, d)), os.listdir(self.base_path))
        for d in directories:
            for d1, subdir, filelist in os.walk(os.path.join(self.base_path, d)):
                print d1
                #print subdir, filter(lambda file_name: fnmatch(file_name, ".svn"), subdir)
                if len(filter(lambda file_name: fnmatch(file_name, ".git"), subdir)) > 0:
                    print "break", d
                    break
                if len(filter(lambda file_name: fnmatch(file_name, ".svn"), subdir)) > 0:
                    print "break2", d
                    break
                self.deletion_list.append(d)
        print self.deletion_list

    def clean_up(self):
        self.retrieve_directory_list()
