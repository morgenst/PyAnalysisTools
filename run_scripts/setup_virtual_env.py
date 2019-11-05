from __future__ import print_function

import argparse
import os
from subprocess import check_call

from builtins import input
from builtins import map
from builtins import object
from builtins import range

from PyAnalysisTools.base import _logger


class SetupHelper(object):
    def __init__(self, **kwargs):
        self.tools = {1: "numpy", 2: "tensorflow", 3: "root_numpy"}
        self.name = kwargs["name"]
        self.install = kwargs["install"]
        self.base_dir = os.path.abspath(kwargs["base_dir"])
        self.bash_script = open(os.path.join(self.base_dir, ".setup_virtual_env.sh"), "w+")
        print("#!/usr/bin/env bash", file=self.bash_script)

    @staticmethod
    def check_virtual_env():
        # try:
        #     import virtualenv
        # except ImportError:
        #     return False
        return True

    def setup_virtual_env(self):
        do_install = False
        user_input = input("virtualenv not installed. Do you want to install? [y|n]")
        if user_input.lower() == "y" or user_input.lower() == "yes":
            do_install = True
        if not do_install:
            exit(0)
        print("", file=self.bash_script)

    def print_menu(self):
        print("Supported applications. Dependencies will be resolved automatically")
        print("1. numpy")
        print("2. tensorflow (Note: Need to be on lxplus7 aka CC7)")
        print("3. root_numpy")
        print("4. sklearn")
        print("5. matplotlib")
        print("6. pandas")
        print("6. python-future")
        print("a. All")

        user_input = input("Please choose (space separated) the tools to install. Hit enter for exit.")
        if user_input == "":
            exit()
        if user_input.lower() == "a" or user_input.lower() == "all":
            tools_to_install = list(range(1, len(self.tools)+1))
        else:
            try:
                tools_to_install = list(map(int, user_input.split()))
            except ValueError:
                _logger.fatal("Invalid input ", user_input)
                return []
        return tools_to_install

    def run_setup(self):
        if not self.check_virtual_env():
            self.setup_virtual_env()
        if os.path.exists(os.path.join(self.base_dir, self.name)):
            print("source {:s}/bin/activate".format(os.path.join(self.base_dir, self.name)), file=self.bash_script)
            return
        print("cd {:s}".format(self.base_dir), file=self.bash_script)
        print("virtualenv {:s}".format(self.name), file=self.bash_script)
        print("source {:s}/bin/activate".format(self.name), file=self.bash_script)
        print("python {:s} self.name --install".format(os.path.abspath(__file__)), file=self.bash_script)

    @staticmethod
    def _install(tool):
        check_call(["pip", "install", "--upgrade", tool])

    def run_install(self):
        tools_to_install = self.print_menu()
        if 1 in tools_to_install:
            self._install("numpy")
        if 2 in tools_to_install:
            self._install("tensorflow")
            self._install("keras")
        if 3 in tools_to_install:
            self._install("root_numpy")
        if 4 in tools_to_install:
            self._install("scikit-learn")
        if 5 in tools_to_install:
            self._install("matplotlib")
        if 6 in tools_to_install:
            self._install("pandas")
        if 7 in tools_to_install:
            self._install("python-future")

    def run(self):
        if not self.install:
            self.run_setup()
        else:
            self.run_install()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Install or setup virtual env")
    parser.add_argument("name", type=str, help="name of virtual env")
    parser.add_argument("--base_dir", "-bd", default="./", help="base directory where to create virtual env")
    parser.add_argument("--install", action="store_true", default=False, help="run installation of packages")

    args = parser.parse_args()
    helper = SetupHelper(**vars(args))
    helper.run()
