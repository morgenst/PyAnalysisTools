import glob
import os
import time

from PyAnalysisTools.base import _logger
from PyAnalysisTools.base.ShellUtils import remove_file, change_dir


class BatchJob(object):
    def __init__(self):
        pass


class BatchHandle(object):
    """
    Class to handle submission and monitoring to batch system
    """
    def __init__(self, job):
        self.system = 'qsub'
        self.queue = job.queue
        self.is_master = False
        self.log_level = job.log_level
        self.local = job.local
        self.job = job
        if job.cluster_cfg_file is None:
            self.is_master = True

    def execute(self):
        _logger.debug("Executing batch job")
        if self.is_master:
            self.job.prepare()
            self.output_dir = self.submit_childs(*self.job.submit_args)
            self.job.finish()
        else:
            self.job.run(self.job.cluster_cfg_file, self.job.job_id)

    def submit_childs(self, cfg_file_name, exec_script, n_jobs, output_dir):
        base_path = os.path.join('/', *os.path.abspath(exec_script).split("/")[1:-2])

        for job_id in range(n_jobs):
            if self.local:
                log_fn = 'log_plotting_{:d}.txt'.format(job_id)
                os.system('python {:s} -mtcf {:s} -id {:d} -log {:s} > {:s}'.format(exec_script, cfg_file_name, job_id,
                                                                                    self.log_level,
                                                                                    os.path.join(output_dir, log_fn)))
                continue
            if 'AnaPySetup' not in os.environ:
                _logger.error('Cannot find env AnaPySetup and thus not determine setup script. Exiting')
                exit(9)
            os.system('echo "source $HOME/.bashrc && cd {:s} && source {:s} && cd macros '
                      '&& python {:s} -mtcf {:s} -id {:d} -log {:s}" | {:s} -q {:s} '
                      '-o {:s}.txt -e {:s}.err'.format(base_path, os.environ['AnaPySetup'], exec_script, cfg_file_name,
                                                       job_id, self.log_level, self.system, self.queue,
                                                       os.path.join(output_dir, 'log_plotting_{:d}'.format(job_id)),
                                                       os.path.join(output_dir, 'log_plotting_{:d}'.format(job_id))))

        while len(glob.glob(os.path.join(output_dir, '*.{:s}'.format('root')))) < n_jobs:
            time.sleep(10)
        return output_dir

    def __del__(self):
        if not self.is_master:
            return
        _logger.debug("Taring log files")
        try:
            with change_dir(self.output_dir):
                os.system('tar -cf {:s} log*.txt log*.err &> /dev/null'.format('logs.tar'))
                for fn in glob.glob('log*.txt'):
                    remove_file(fn)
                for fn in glob.glob('log*.err'):
                    remove_file(fn)
        except AttributeError:
            _logger.error('Something went wrong in processing. Could not clean up. Check carefully')
