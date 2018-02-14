import argparse
import os
import sys


class SetupHelper(object):
    def __init__(self, name):
        self.bash_script = open(".setup_virtual_env.sh", "w+")
        self.tools = {1: "numpy", 2: "tensorflow", 3: "root_numpy"}
        self.name = name
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
            tools_to_install = range(len(self.tools))
        else:
            tools_to_install = map(int, user_input.split())
        return tools_to_install

    def run(self):
        if not self.check_virtual_env():
            self.setup_virtual_env()
        if os.path.exists(self.name):
            print >> self.bash_script, "source {:s}/bin/activate".format(self.name)
            return
        print >> self.bash_script, "virtualenv {:s}".format(self.name)
        print >> self.bash_script, "source {:s}/bin/activate".format(self.name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Install or setup virtual env")
    parser.add_argument("name", type=str, help="name of virtual env")

    args = parser.parse_args()
    helper = SetupHelper(name=args.name)
    helper.run()
