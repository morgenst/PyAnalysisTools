import shutil
import os
import subprocess


def make_dirs(path):
    if os.path.exists(path):
        return
    try:
        os.makedirs(path)
    except OSError as e:
        raise OSError


def resolve_path_from_symbolic_links(symbolic_link, relative_path):
    if symbolic_link is None:
        return relative_path
    if not os.path.islink(symbolic_link) or os.path.isabs(relative_path):
        return relative_path
    return os.path.abspath(os.path.join(symbolic_link, relative_path))


def move(src, dest):
    try:
        shutil.move(src, dest)
    except IOError:
        raise


def copy(src, dest):
    try:
        shutil.copy(src, dest)
    except:
        raise


def remove_directory(path, safe=False):
    if safe:
        try:
            os.removedirs(path)
        except OSError:
            raise
    else:
        shutil.rmtree(path)


def source(script_name):
    print subprocess.PIPE
    pipe = subprocess.Popen(". %s; env" % script_name, stdout=subprocess.PIPE, shell=True)
    output = pipe.communicate()[0]
    output = filter(lambda l: len(l.split("=")) == 2, output.splitlines())
    env = dict((line.split("=", 1) for line in output))
    os.environ.update(env)