from __future__ import print_function

import argparse
import re
import os
import pkg_resources


def check_dependency(requirement):
    try:
        pkg_resources.require(requirement)
    except (pkg_resources.VersionConflict, pkg_resources.DistributionNotFound):
        print("Could not find {:s}. Consider running pip install --user {:s}".format(requirement,
                                                                                     requirement.split(' ')[0]))


if __name__ == '__main__':
    """
    Recursively confirm that requirements are available.
    """

    parser = argparse.ArgumentParser(description='check availability of python dependencies')
    parser.add_argument('--basic', action='store_true', default=False, help='check only basic packages for C++ usage')

    args = parser.parse_args()

    if args.basic:
        for req in ['future', 'oyaml']:
            check_dependency(req)
    else:
        req_file = os.path.join(os.path.dirname(__file__), 'requirements.txt')
        with open(req_file, 'r') as f:
            requirements = f.readlines()
            requirements = [r.strip() for r in requirements]
            requirements = [r for r in sorted(requirements) if r and not r.startswith('#')]

            excl_packages = ['coverage', 'nose', 'flake*', 'mock']
            excl_packages_regex = "(" + ")|(".join(excl_packages) + ")"
            for requirement in requirements:
                if re.match(excl_packages_regex, requirement):
                    continue
            check_dependency(requirement)
