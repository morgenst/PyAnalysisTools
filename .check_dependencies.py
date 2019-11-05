from __future__ import print_function
import os
import pkg_resources

if __name__ == '__main__':
    """
    Recursively confirm that requirements are available.
    """

    req_file = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    with open(req_file, 'r') as f:
        requirements = f.readlines()
        requirements = [r.strip() for r in requirements]
        requirements = [r for r in sorted(requirements) if r and not r.startswith('#')]
        for requirement in requirements:
            try:
                pkg_resources.require(requirement)
            except pkg_resources.VersionConflict:
                print("Could not find {:s}. Consider running pip install -u {:s}".format(requirement,
                                                                                         requirement.split(' ')[0]))
