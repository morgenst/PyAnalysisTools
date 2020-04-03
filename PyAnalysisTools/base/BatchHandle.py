from __future__ import print_function

import glob
import os
import time
import stat
from getpass import getuser
from random import randint
from subprocess import check_output

from PyAnalysisTools.base import _logger
import dill

from PyAnalysisTools.base.ShellUtils import change_dir, remove_file

dill._dill._reverse_typemap['ObjectType'] = object


class BatchJob(object):
    def __init__(self):
        pass


class BatchHandle(object):
    """
    Class to handle submission and monitoring to batch system
    """

    def __init__(self, job):
        if not hasattr(job, 'log_fname'):
            job.log_fname = 'log'
        if not hasattr(job, 'identifier'):
            job.identifier = 'batch_{:d}'.format(randint(0, 1000))
        self.system = 'qsub'
        if hasattr(job, 'extra_submission_args'):
            if job.extra_submission_args is not None:
                self.system += ' {:s}'.format(job.extra_submission_args)

        self.queue = job.queue
        self.is_master = False
        self.identifier = job.identifier
        self.log_level = job.log_level
        self.log_fname = job.log_fname
        self.local = job.local
        self.job = job
        if job.cluster_cfg_file is None:
            self.is_master = True

    def execute(self):
        _logger.debug("Executing batch job")
        if self.is_master:
            job_ids = self.job.prepare()
            self.output_dir = self.submit_childs(*self.job.submit_args, job_ids=job_ids)
            self.job.finish()
        else:
            self.job.run(self.job.cluster_cfg_file, self.job.job_id)

    def submit_childs(self, cfg_file_name, exec_script, n_jobs, output_dir, disable_wait=False, job_ids=None):
        base_path = os.path.join('/', *os.path.abspath(exec_script).split("/")[1:-2])
        _logger.debug('childs ids: {} {:d}'.format(str(job_ids), n_jobs))
        for job_id in range(n_jobs):
            if job_ids is not None and job_id not in job_ids:
                continue
            if self.local:
                log_fn = '{:s}_{:d}.txt'.format(self.log_fname, job_id)
                # os.system('python {:s} -mtcf {:s} -id {:d} -log {:s}'.format(exec_script, cfg_file_name, job_id,
                #                                                                     self.log_level))
                os.system('python {:s} -mtcf {:s} -id {:d} -log {:s} > {:s}'.format(exec_script, cfg_file_name, job_id,
                                                                                    self.log_level,
                                                                                    os.path.join(output_dir, log_fn)))
                continue
            if 'AnaPySetup' not in os.environ:
                _logger.error('Cannot find env AnaPySetup and thus not determine setup script. Exiting')
                exit(9)
            bash_script = os.path.join(os.path.join(output_dir, 'submit_{:s}_{:d}.sh'.format(self.identifier,
                                                                                             job_id)))
            with open(bash_script, 'wb') as f:
                print('#!/usr/bin/env bash', file=f)
                print('source $HOME/.bashrc && cd {:s} && source {:s} && cd macros && python {:s} -mtcf {:s} -id {:d} '
                      '-log {:s}'.format(base_path, os.environ['AnaPySetup'], exec_script, cfg_file_name, job_id,
                                         self.log_level), file=f)
            os.chmod(bash_script, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IWGRP |
                     stat.S_IXGRP | stat.S_IROTH)
            log_file_name = os.path.join(output_dir, '{:s}_{:d}'.format(self.log_fname, job_id))
            os.system('{:s} {:s} -q {:s} -o {:s}.txt -e {:s}.err'.format(self.system, bash_script, self.queue,
                                                                         log_file_name, log_file_name))
            time.sleep(2)
        while check_output(['qstat', '-u {:s}'.format(getuser())]).count('submit_{:s}'.format(self.identifier)) > 0:
            time.sleep(10)
        if not disable_wait:
            n_processed_files = len(glob.glob(os.path.join(output_dir, '*.{:s}'.format('root'))))
            if n_processed_files < n_jobs:
                _logger.error("Could not produce all root files. Expect {:d}, but found "
                              "only {:d}. Giving up, not running post-processing".format(n_jobs, n_processed_files))
                exit(-1)
        return output_dir

    def __del__(self):
        if not self.is_master:
            return
        _logger.debug("Taring log files")
        time.sleep(120)
        try:
            with change_dir(self.output_dir):
                os.system('tar -cf {:s} log*.txt log*.err &> /dev/null'.format('logs.tar'))
                os.system('tar -cf {:s} submit*.sh &> /dev/null'.format('scripts.tar'))
                for fn in glob.glob('log*.txt'):
                    remove_file(fn)
                for fn in glob.glob('log*.err'):
                    remove_file(fn)
                for fn in glob.glob('submit*.sh'):
                    remove_file(fn)
        except AttributeError:
            _logger.error('Something went wrong in processing. Could not clean up. Check carefully')
