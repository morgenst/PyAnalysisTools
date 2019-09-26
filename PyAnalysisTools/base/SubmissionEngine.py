import os
import shutil
from PyAnalysisTools.base import InvalidInputError
from PyAnalysisTools.base.ShellUtils import make_dirs
from subprocess import check_call


class JobConfig(object):
    """
    Simple configuration wrapper class
    """
    def __init__(self, **kwargs):
        kwargs.setdefault('queue', 'generic')

        for k, v in kwargs.iteritems():
            setattr(self, k, v)


class SubmissionEngine(object):
    def __init__(self, **kwargs):
        kwargs.setdefault('file_to_copy', [])
        kwargs.setdefault('file_to_copy_path', None)
        kwargs.setdefault('log_file_name', None)
        kwargs.setdefault('eff_file_name', None)
        kwargs.setdefault('callee_dir', None)

        if 'output_path' not in kwargs:
            raise InvalidInputError('Missing output path. Need to give up.')

        for k, v in kwargs.iteritems():
            setattr(self, k, v)

    def prepare(self):
        if self.file_to_copy is not None:
            self.abs_files_to_copy = map(lambda fn: os.path.join(os.path.abspath(self.callee_dir), fn),
                                         self.files_to_copy)

    def prepare_job(self):
        make_dirs(self.output_path)
        for fn in self.files_to_copy:
            shutil.copy(os.path.join(os.path.abspath(self.callee_dir), fn),
                        os.path.join(self.output_path, fn))

    def submit(self, **job_args):
        job_args = JobConfig(**job_args)
        abs_job_output_path = os.path.join(self.output_path, job_args.job_output_path)
        log_file = os.path.join(abs_job_output_path, self.log_file_name)
        err_file = os.path.join(abs_job_output_path, self.err_file_name)
        make_dirs(abs_job_output_path)
        job_args.parameters['JOP'] = abs_job_output_path
        check_call(['qsub', '-q', job_args.queue, '-o', log_file, '-e', err_file,
                    os.path.join(os.path.abspath(self.callee_dir), job_args.run_script),
                    '-v', str(','.join(['{:s}={:s}'.format(opt, val) for opt, val in job_args.parameters.iteritems()]))])
