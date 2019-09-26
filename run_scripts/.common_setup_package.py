import os
import re
import yaml
import sys
config_file_name = ".PACKAGE_NAME_LOWER_config.yml"


def expand_variables():
    f = open(config_file_name, "r")
    data = yaml.load(f)
    f.close()
    f_setup = open(".set_env.sh", "w")
    print >> f_setup, "#!/bin/bash"
    for arg, val in data.iteritems():
        print >> f_setup, "export {}={}".format(arg, val)
    f_setup.close()


def initialise_yaml():
    data = {}
    print
    print
    print "-" * 50
    print "setting up PACKAGE_NAME"
    f = open(config_file_name, "w")
    user_name = raw_input("Please enter grid user name: ")
    if re.match(r"^[a-zA-Z]+$", user_name) is None:
        print "Invalid user name: ", user_name, ". Must be string."
        f.close()
        os.remove(config_file_name)
        sys.exit(1)
    nice_user_name = raw_input("Please enter your nice user name (aka cern user name): ")
    if re.match(r"^[a-zA-Z]+$", nice_user_name) is None:
        print "Invalid user name: ", nice_user_name, ". Must be string."
        f.close()
        os.remove(config_file_name)
        sys.exit(1)
    data["PACKAGE_NAME_ABBR_NICKNAME"] = user_name
    data["PACKAGE_NAME_ABBR_NICENAME"] = nice_user_name
    yaml.dump(data, f)
    f.close()
    

if __name__ == '__main__':
    if not os.path.exists(config_file_name):
        initialise_yaml()
    expand_variables()
