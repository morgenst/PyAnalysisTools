import shutil
import os
import subprocess


def make_dirs(path):
    path = os.path.expanduser(path)
    if os.path.exists(path):
        return
    try:
        os.makedirs(path)
    except OSError as e:
        raise OSError


def resolve_path_from_symbolic_links(symbolic_link, relative_path):
    def is_symbolic_link(path):
        return os.path.islink(path)
    if symbolic_link is None or relative_path is None:
        return relative_path
    if os.path.isabs(relative_path):
        return relative_path
    if not symbolic_link.endswith("/"):
        symbolic_link += "/"
    top_level_dir = symbolic_link.split("/")
    for n in range(1, len(top_level_dir)):
        if is_symbolic_link("/".join(top_level_dir[:-n])):
            return os.path.abspath(os.path.join(symbolic_link, relative_path))
    return relative_path


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
    if not os.path.exists(path):
        return
    if safe:
        try:
            os.removedirs(path)
        except OSError:
            raise
    else:
        try:
            shutil.rmtree(path)
        except OSError as e:
            raise e


def source(script_name):
    pipe = subprocess.Popen(". %s; env" % script_name, stdout=subprocess.PIPE, shell=True)
    output = pipe.communicate()[0]
    output = filter(lambda l: len(l.split("=")) == 2, output.splitlines())
    env = dict((line.split("=", 1) for line in output))
    os.environ.update(env)
