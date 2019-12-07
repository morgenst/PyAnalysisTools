import ROOT
import os
from subprocess import check_call
from functools import partial
from multiprocessing import Pool
from PyAnalysisTools.base.ShellUtils import move, remove_directory, make_dirs


class Writer:
    def __init__(self, directory=None):
        """

        :param directory:
        """
        if directory is None:
            directory = os.path.abspath(os.curdir)
        self.dir = directory
        self.__check_and_create_directory(self.dir)

    def __check_and_create_directory(self, directory):
        _logger.debug("Check if directory: %s exists" % (directory))
        if not os.path.exists(directory):
            _logger.debug("Create directory: %s exists" % (directory))
            os.makedirs(directory)

    def dump_canvas(self, canvas, message=None, image=None):
        if image:
            self.write_canvas_to_file(canvas, image)
        else:
            if message is None:
                image = raw_input("save canvas as (<RET> for skipping): ")
            else:
                image = raw_input(message)

            if image:
                self.write_canvas_to_file(canvas, image)

    def write_canvas_to_file(self, canvas, name, extension='pdf'):
        ext = self.parse_extension_from_file_name(name)
        if ext is not None:
            extension = ext
            name = ''.join(name.split('.')[0:-1])
        if not extension.startswith('.'):
            extension = '.' + extension
        if extension == '.root':
            self.write_object_to_root_file(canvas, name + extension)
        else:
            canvas.SaveAs(os.path.join(os.path.join(self.dir,
                                                    name + extension)))

    def write_object_to_root_file(self, obj, filename, dir=''):
        f = ROOT.gROOT.GetListOfFiles().FindObject(filename)
        if not f:
            f = ROOT.TFile.Open(filename, 'UPDATE')
        d = f.GetDirectory(dir)
        if not d:
            d = make_root_dir(f, dir)
        d.cd()
        obj.Write()

    @staticmethod
    def parse_extension_from_file_name(name):
        ext = name.split('.')[-1]
        if ext is name:
            return None
        return ext

    def set_directory(self, directory):
        self.__check_and_create_directory(directory)
        self.dir = directory


def parallel_merge(data, output_path, prefix, merge_dir=None, force=False, postfix=None, ncpu=10):
    make_dirs(output_path)
    if len(os.listdir(merge_dir)) > 0:
        do_delete = raw_input("Merge directory contains already files. Shall I delete those?: [y|n]")
        if do_delete.lower() == "y" or do_delete.lower() == "yes":
            map(lambda d: remove_directory(os.path.join(merge_dir, d)), os.listdir(merge_dir))

    pool = Pool(processes=min(ncpu, len(data)))
    for item in data.items():
        parallel_merge_wrapper(item, output_path=output_path, prefix=prefix,
                               merge_dir=merge_dir, force=force, postfix=postfix), data.items()
    # pool.map(partial(parallel_merge_wrapper, output_path=output_path, prefix=prefix,
    #                  merge_dir=merge_dir, force=force, postfix=postfix), data.items())


def parallel_merge_wrapper(dict_element, output_path, prefix, merge_dir=None, force=False, postfix=None):
    process, input_file_list = dict_element
    if merge_dir is not None:
        merge_dir = os.path.join(merge_dir, process)
    merge_files(input_file_list, output_path, prefix + "{:s}".format(process), merge_dir, force, postfix)


def merge_files(input_file_list, output_path, prefix, merge_dir=None, force=False, postfix=None):
    def build_buckets(file_list):
        limit = 2. * 1024. * 1024. * 1024.
        if sum(map(os.path.getsize, file_list)) < limit:
            return [file_list]
        bucket_list = []
        tmp = []
        summed_file_size = 0.
        for file_name in file_list:
            if summed_file_size > limit:
                summed_file_size = 0.
                bucket_list.append(tmp)
                tmp = []
            summed_file_size += os.path.getsize(file_name)
            tmp.append(file_name)
        bucket_list.append(tmp)
        return bucket_list

    def merge(file_lists):
        print os.path.abspath(os.curdir)
        import time
        time.sleep(2)
        if len([f for chunk in file_lists for f in chunk]) == 0:
            return
        for file_list in file_lists:
            merge_cmd = 'hadd '
            if force:
                merge_cmd += ' -f '
            if postfix is not None:
                output_file_name = '{:s}_{:d}.{:s}.root'.format(prefix, file_lists.index(file_list), postfix)
            else:
                output_file_name = '{:s}_{:d}.root'.format(prefix, file_lists.index(file_list))
            merge_cmd += '%s %s' % (output_file_name, ' '.join(file_list))
            #print merge_cmd
            if not force and os.path.exists(os.path.join(output_path, output_file_name)):
                continue
            #check_call(merge_cmd.split())
            os.system(merge_cmd)
            if not merge_dir == output_path:
                move(output_file_name, os.path.join(output_path, output_file_name))

    def setup_paths(merge_dir):
        if not os.path.exists(output_path):
            make_dirs(output_path)
        if merge_dir is None:
            merge_dir = output_path
        else:
            merge_dir = os.path.abspath(merge_dir)
            make_dirs(merge_dir)
        os.chdir(merge_dir)

    buckets = build_buckets(input_file_list)
    setup_paths(merge_dir)
    #merge(buckets, prefix, output_path, merge_dir, force)
    merge(buckets)
    if merge_dir is not None:
        remove_directory(os.path.abspath(merge_dir))
