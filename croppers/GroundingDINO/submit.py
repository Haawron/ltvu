import submitit
from pathlib import Path
from functools import partial

from hello import main


def submit_job_to_slurm() -> list[submitit.Job]:
    world_size = 32

    p_logs = Path() / f'logs/%A/%j'
    executor = submitit.AutoExecutor(folder=p_logs)
    gpus_per_node = 1  # 그냥 job 당 노드
    executor.update_parameters(
        slurm_job_name=f'G-DINO_EgoNLQ_bbox_detection',
        gpus_per_node=gpus_per_node,
        tasks_per_node=1,
        cpus_per_task=8,
        mem_gb=29*gpus_per_node,
        nodes=1,
        timeout_min=60*24*6,  # Max time: 6 days
        slurm_partition="batch_grad",
        slurm_array_parallelism=world_size,
        slurm_additional_parameters=dict(
            chdir='/data/gunsbrother/repos/GroundingDINO',
            exclude='ariel-k1',
        )
    )

    # Modify these arguments as per your requirement
    jobs = executor.map_array(partial(main, world_size=world_size), list(range(world_size)))  # saves function and arguments in a pickle file
    return jobs


if __name__ == "__main__":
    jobs = submit_job_to_slurm()
    for job in jobs:
        print(f"Submitted job with ID: {job.job_id}")
    for job in jobs:
        print(job, job.result())
