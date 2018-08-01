import shutil
import os
import subprocess
import sys
from contextlib import contextmanager


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


def fileno(file_or_fd):
    fd = getattr(file_or_fd, 'fileno', lambda: file_or_fd)()
    if not isinstance(fd, int):
        raise ValueError("Expected a file (`.fileno()`) or a file descriptor")
    return fd


@contextmanager
def std_stream_redirected(dest=os.devnull, stream=sys.stdout, std_stream=None):
    if std_stream is None:
       std_stream = stream

    std_stream_fd = fileno(std_stream)

    with os.fdopen(os.dup(std_stream_fd), 'wb') as copied:
        stream.flush()
        try:
            os.dup2(fileno(dest), std_stream_fd)
        except ValueError:
            with open(dest, 'wb') as to_file:
                os.dup2(to_file.fileno(), std_stream_fd)
        try:
            yield stream
        finally:
            stream.flush()
            os.dup2(copied.fileno(), std_stream_fd)
