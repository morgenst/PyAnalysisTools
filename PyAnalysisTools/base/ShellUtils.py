import glob
import shutil
import os
import subprocess
import sys
from contextlib import contextmanager
from threading import Lock
from PyAnalysisTools.base import _logger


def make_dirs(path):
    """
    Create (nested) directories (thread-safe)
    :param path: directory
    :type path: str
    :return: Nothing
    :rtype: None
    """
    lock = Lock()
    lock.acquire()
    path = os.path.expanduser(path)
    if os.path.exists(path):
        return
    try:
        os.makedirs(path)
    except OSError:
        _logger.error('Unable to create directory {:s}'.format(path))
        raise OSError
    finally:
        lock.release()


def resolve_path_from_symbolic_links(symbolic_link, relative_path):
    """
    Expand symbolic link in relative path. Needed to deal with sysmlinks to eos
    :param symbolic_link: input link
    :type symbolic_link: str
    :param relative_path: name of relative path
    :type relative_path: str
    :return: relative path w.r.t symbolic link
    :rtype: str
    """
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
    """
    Wrapper of OS move operation.
    :param src: input source
    :type src: string
    :param dest: destination
    :type dest: str
    :return: Nothing
    :rtype: None
    """
    if '*' in src:
        for fn in glob.glob(src):
            move(fn, dest)
    else:
        try:
            shutil.move(src, dest)
        except IOError as e:
            raise e


def copy(src, dest):
    """
    Wrapper for OS copy operation.
    :param src: source
    :type src: string
    :param dest: destination
    :type dest: str
    :return: Nothing
    :rtype: None
    """
    try:
        shutil.copy(src, dest)
    except IOError:
        shutil.copytree(src, dest)
    except:
        raise


def remove_directory(path, safe=False):
    """
    Delete directory and its contents
    :param path: input path to be deleted
    :type path: string
    :param safe: switch to check if directory is empty
    :type safe: bool
    :return: Nothing
    :rtype: None
    """
    if not os.path.exists(path):
        return
    if safe:
        try:
            os.removedirs(path)
        except OSError as e:
            raise e
    else:
        try:
            shutil.rmtree(path)
        except OSError as e:
            raise e


def remove_file(file_name):
    try:
        os.remove(file_name)
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


def find_file(file_name, subdirectory=''):
    if subdirectory:
        path = subdirectory
    else:
        path = os.getcwd()
    for root, dirs, names in os.walk(path):
        if file_name in names:
            return os.path.join(root, file_name)
    return None


@contextmanager
def change_dir(path):
    """
    Custom change dir. Changes to path, executes and returns to old path
    :param path:
    :return:
    """
    oldpwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(oldpwd)
