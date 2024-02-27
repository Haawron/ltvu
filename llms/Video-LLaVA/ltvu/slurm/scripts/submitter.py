import time
import logging
from pathlib import Path
from datetime import datetime
from collections.abc import Callable

import submitit


class Submitter:
    def __init__(self,
        model_name = 'Video-LLAVA-captioning',
        run_name = '',
        cpus_per_gpu = 8,
        mem_gb_per_gpu = 29,
        slurm_partition = 'batch_grad',
        slurm_array_parallelism = 16,
        slurm_additional_parameters = dict(
            # chdir='',
            exclude='ariel-k[1-2]',
        ),
    ):
        self.timestamp = datetime.now().strftime(r'%y%m%d-%H%M%S')
        self.model_name = model_name
        self.run_name = run_name
        self.cpus_per_gpu = cpus_per_gpu
        self.mem_gb_per_gpu = mem_gb_per_gpu
        self.slurm_partition = slurm_partition
        self.slurm_array_parallelism = slurm_array_parallelism
        self.slurm_additional_parameters = slurm_additional_parameters

        p_slurm_dir = Path(__file__).parent.parent
        if self.run_name:
            self.exp_name = f'{self.model_name}-{self.run_name}'
            self.template_p_logs_dir = p_slurm_dir / f'logs/%A-{self.run_name}/%j'  # %A means the jobname
        else:
            self.exp_name = f'{self.model_name}'
            self.template_p_logs_dir = p_slurm_dir / f'logs/%A/%j'

        print(f'Experiment name: {self.exp_name}')
        print(f'Logs dir template: {self.template_p_logs_dir.absolute()}')
        print()

    def submit_jobs(self,
        target_func: Callable,
        arguments: list[dict]|list[list] = [{}],
        **additional_slurm_parameters,
    ) -> list[submitit.Job]:
        executor = submitit.AutoExecutor(folder=self.template_p_logs_dir)
        slurm_parameters = self._get_slurm_paramters()
        slurm_parameters.update(additional_slurm_parameters)
        executor.update_parameters(**slurm_parameters)

        # Submit: saves the target function and arguments in a pickle file
        # len(arguments) will be the number of runs
        if isinstance(arguments[0], dict):
            jobs = executor.map_array(target_func, arguments)
        elif isinstance(arguments[0], list):
            jobs = executor.map_array(target_func, *arguments)
        else:
            jobs = executor.map_array(target_func, arguments)
        for job in jobs:
            print(f"Submitted a job with ID: {job.job_id}")

        return jobs

    def _get_slurm_paramters(self):
        gpus_per_node = 1
        slurm_parameters = dict(
            slurm_job_name=self.exp_name,
            gpus_per_node=gpus_per_node,
            tasks_per_node=1,
            cpus_per_task=self.cpus_per_gpu,
            mem_gb=self.mem_gb_per_gpu * gpus_per_node,
            nodes=1,
            timeout_min=60*24*6,  # Max time: 6 days
            slurm_partition=self.slurm_partition,
            slurm_array_parallelism=self.slurm_array_parallelism,
            slurm_additional_parameters=self.slurm_additional_parameters,
        )
        return slurm_parameters


class InfoOnlyFilter(logging.Filter):
    def filter(self, record):
        return record.levelno == logging.INFO


class PackagePathFilter(logging.Filter):
    def filter(self, record):
        pathname = record.pathname
        record.relativepath = str(Path(pathname).relative_to(Path().absolute()))
        return True


def get_logger(name='array_job', p_log_dir=None):
    logger = logging.getLogger(name=name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(relativepath)s:%(funcName)s %(levelname)s (%(asctime)s) - %(message)s')

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.addFilter(InfoOnlyFilter())
    console_handler.addFilter(PackagePathFilter())
    logger.addHandler(console_handler)

    if p_log_dir:
        p_log = p_log_dir / 'array_jobs.log'
        p_log_err = p_log.with_suffix('.err')

        stdout_handler = logging.FileHandler(p_log)
        stdout_handler.setFormatter(formatter)
        stdout_handler.addFilter(InfoOnlyFilter())
        stdout_handler.addFilter(PackagePathFilter())
        logger.addHandler(stdout_handler)

        stderr_handler = logging.FileHandler(p_log_err)
        stderr_handler.setFormatter(formatter)
        stderr_handler.setLevel(logging.ERROR)
        stderr_handler.addFilter(PackagePathFilter())
        logger.addHandler(stderr_handler)

    return logger


def wait(jobs: list[submitit.Job]):
    N = len(jobs)
    w = len(str(N))
    print()
    print(f'Waiting for {N} jobs finishing ...')
    completed = set()
    logger = get_logger('Array job full', Path(jobs[0].paths.folder).parent)
    no_error_occurred = True
    while len(completed) < N:
        for i, job in enumerate(jobs):
            if i not in completed and job.done():
                log_string_base = f'[{len(completed)+1:{w}d}/{N:{w}d}] {job.job_id}: '
                try:
                    result = job.result()
                except Exception as e:
                    log_string = log_string_base + str(e)
                    logger.error(log_string)
                    result = f'Got an error {job.paths.stderr}'
                    no_error_occurred = False
                finally:
                    log_string = log_string_base + str(result)
                    logger.info(log_string)
                completed.add(i)
        time.sleep(1)  # Sleep to avoid constant polling
    return no_error_occurred
