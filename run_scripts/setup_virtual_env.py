import argparse
import os
import sys
from PyAnalysisTools.base import _logger
from subprocess import check_call


class SetupHelper(object):
    def __init__(self, **kwargs):
        self.tools = {1: "numpy", 2: "tensorflow", 3: "root_numpy"}
        self.name = kwargs["name"]
        self.install = kwargs["install"]
        self.base_dir = os.path.abspath(kwargs["base_dir"])
        self.bash_script = open(os.path.join(self.base_dir, ".setup_virtual_env.sh"), "w+")

        print >> self.bash_script, "#!/usr/bin/env bash"

    @staticmethod
    def check_virtual_env():
        try:
            import virtualenv
        except ImportError:
            return False
        return True

    def setup_virtual_env(self):
        do_install = False
        user_input = raw_input("virtualenv not installed. Do you want to install? [y|n]")
        if user_input.lower() == "y" or user_input.lower() == "yes":
            do_install = True
        if not do_install:
            exit(0)
        print >> self.bash_script, ""

    def print_menu(self):
        print "Supported applications. Dependencies will be resolved automatically"
        print "1. numpy"
        print "2. tensorflow (Note: Need to be on lxplus7 aka CC7)"
        print "3. root_numpy"
        print "a. All"

        user_input = raw_input("Please choose (space separated) the tools to install. Hit enter for exit.")
        if user_input == "":
            exit()
        if user_input.lower() == "a" or user_input.lower() == "all":
            tools_to_install = range(1,len(self.tools)+1)
        else:
            try:
                tools_to_install = map(int, user_input.split())
            except ValueError:
                _logger.fatal("Invalid input ", user_input)
                return []
        return tools_to_install

    def run_setup(self):
        if not self.check_virtual_env():
            self.setup_virtual_env()
        if os.path.exists(os.path.join(self.base_dir, self.name)):
            print >> self.bash_script, "source {:s}/bin/activate".format(os.path.join(self.base_dir, self.name))
            return
        print >> self.bash_script, "cd {:s}".format(self.base_dir)
        print >> self.bash_script, "virtualenv {:s}".format(self.name)
        print >> self.bash_script, "source {:s}/bin/activate".format(self.name)
        print >> self.bash_script, "python {:s} self.name --install".format(os.path.abspath(__file__))

    @staticmethod
    def _install(tool):
        check_call(["pip", "install", "--upgrade", tool])

    def run_install(self):
        tools_to_intall = self.print_menu()
        if 1 in tools_to_intall:
            self._install("numpy")
        if 2 in tools_to_intall:
            self._install("tensorflow")
            self._install("keras")
        if 3 in tools_to_intall:
            self._install("root_numpy")

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
