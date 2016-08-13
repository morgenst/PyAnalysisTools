import shutil
import os


def make_dirs(path):
    if os.path.exists(path):
        return
    try:
        os.makedirs(path)
    except OSError as e:
        raise OSError


def resolve_path_from_symbolic_links(symbolic_link, relative_path):
    if not os.path.islink(symbolic_link) or os.path.isabs(relative_path):
        return
    return os.path.abspath(os.path.join(symbolic_link, relative_path))


def move(src, dest):
    try:
        shutil.move(src, dest)
    except IOError:
        raise


def remove_directory(path, safe=False):
    if safe:
        try:
            os.removedirs(path)
        except OSError:
            raise
    else:
        shutil.rmtree(path)