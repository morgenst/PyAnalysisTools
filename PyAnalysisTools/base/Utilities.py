import os
import calendar
import time
import fnmatch
from PyAnalysisTools.base import _logger, InvalidInputError
from PyAnalysisTools.base.YAMLHandle import YAMLLoader
from PyAnalysisTools.base import ShellUtils


def merge_dictionaries(*dicts):
    result = {}
    for dictionary in dicts:
        result.update(dictionary)
    return result


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


def check_required_args(*args, **kwargs):
    for arg in args:
        if arg not in kwargs:
            return arg
    return None


class Cleaner(object):
    def __init__(self, **kwargs):
        if "base_path" not in kwargs:
            _logger.error("No path provided")
            raise InvalidInputError("No path provided.")

        kwargs.setdefault("safe", True)
        self.base_path = os.path.abspath(kwargs["base_path"])
        self.safe = kwargs["safe"]
        self.keep_pattern = [".git", ".keep", ".svn", "InstallArea", "RootCoreBin", "WorkArea"]
        self.deletion_list = []
        self.touch_threshold_days = 14.
        self.trash_path = os.path.expanduser(kwargs["trash_path"])

    def setup_temporary_trash(self):
        if not self.safe:
            return
        ShellUtils.make_dirs(self.trash_path)

    @staticmethod
    def check_lifetime(threshold, d1, filelist):
        return divmod(calendar.timegm(time.gmtime()) - os.path.getctime(min(map(lambda fn: os.path.join(d1, fn),
                                                                                filelist), key=os.path.getctime)),
                      3600. * 24.)[0] < threshold

    def retrieve_directory_list(self):
        directories = filter(lambda d: os.path.isdir(os.path.join(self.base_path, d)), os.listdir(self.base_path))
        keep_list = []
        for d in directories:
            for d1, subdir, filelist in os.walk(os.path.join(self.base_path, d)):
                if len(filter(lambda cd: d1.startswith(cd), keep_list)) > 0:
                    continue
                if len(filelist):
                    try:
                        if self.check_lifetime(self.touch_threshold_days, d1, filelist):
                            continue
                    except OSError:
                        # todo: check for symlinks, remove invalid ones and check again
                        continue
                self.deletion_list.append(d1)
                if any(keep_pattern in d1 for keep_pattern in self.keep_pattern):
                    keep_list.append(d1)
                if len(filter(lambda sd: sd in self.keep_pattern, subdir)) > 0:
                    keep_list.append(d1)
                if len(filter(lambda fn: fn in self.keep_pattern, filelist)) > 0:
                    keep_list.append(d1)

        self.deletion_list = filter(lambda dn: True not in map(lambda keep: dn.startswith(keep) or keep.startswith(dn),
                                                               keep_list),
                                    self.deletion_list)
        self.deletion_list = filter(lambda dn: True not in map(lambda base: dn.startswith(base) and dn != base,
                                                               self.deletion_list), self.deletion_list)

    def clear_trash(self):
        for d1, subdir, filelist in os.walk(self.trash_path):
            if len(filelist) == 0:
                continue
            if self.check_lifetime(self.touch_threshold_days * 2, d1, filelist):
                ShellUtils.remove_directory(d1)

    def move_to_trash(self):
        for item in self.deletion_list:
            ShellUtils.move(item, self.trash_path)

    def clean_up(self):
        self.retrieve_directory_list()
        self.setup_temporary_trash()
        self.clear_trash()
        self.move_to_trash()
