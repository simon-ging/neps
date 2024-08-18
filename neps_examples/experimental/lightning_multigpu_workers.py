"""
Tested environment: ubuntu, latest torch as of 2024-08, slurm cluster, 2xRTX2080Ti

Usage:
    conda create -n nepstest python=3.10 -y
    conda activate nepstest
    pip install torch torchvision lightning neps neural-pipeline-search
    # get a slurm job with 2 gpus (or change NUM_GPUS below) in foreground
    # ---------- do one eval
    python lightning_multigpu_workers.py
    # ---------- do evals until max_evaluations_total is reached, or an uncaught error happens
    while true; do
        echo -e "\n ----------- new config\n"
        python lightning_multigpu_workers.py
        RETCODE=$?
        if [ $RETCODE -ne 0 ]; then
            echo -e "\n---------- exiting with code $RETCODE\n"
            break
        fi
        sleep 5
        echo
    done
    # test sbatch script usage. adopt the file to your environment first.
    sbatch lightning_multigpu_workers.sh

problem in running neps with one worker having multiple gpus:
1. neps.run samples a config, then calls the training script.
2. the training script uses DDP and spawns more processes for the gpus
3. each instance registers with neps.run as a new worker and gets a new config

solution is to have only rank 0 communicate with neps, and the other ranks find out which
config rank 0 is running, and then run that same config.
in regular ddp setting with e.g. torchrun this can be solved e.g. by broadcasting the config number.

however in lightning there is no process group until trainer.fit is called so there is no broadcast.
so instead we use the SLURM_JOB_ID to find a common path on the filesystem for this neps worker.
then rank 0 communicates with neps, gets a config, and dumps it to the common path,
then calls trainer.fit. other ranks ignore neps, find the common path, and call the trainer.fit
with the same trainer configuration. finally we kill everything after one run for safety.

1. find common identifier over all gpus, using slurm jobid and hpo project name.
2. make sure there is no rank0_info_file with that identifier.
3. write the current config to that file
4. start the trainer. the trainer spawns all the other processes.
5. the other processes find out they are not rank 0, so they load the rank0_info_file and
   start the trainer with the same config.
6. main deletes the rank0_info_file after the run.
7. in order to avoid any complications with ddp, set max_evaluations_per_run=1 and let
   all processes end after one run. then simply restart the script to do another run.
   it might work to keep the processes and create a new trainer with the same number of gpus
   but it might also create subtle bugs that are difficult to spot.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import lightning as L
import yaml
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

import neps
from neps.utils.common import get_initial_directory
from neps.utils.run_args import MAX_EVALUATIONS_TOTAL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NUM_GPUS = 2


def main():
    process_group_key = get_process_group_key()
    print_with_rank(f"{process_group_key=}")
    base_dir = Path.home() / f".cache/neps_debug/{process_group_key}"

    # check whether we reached max evaluations
    neps_config_file = Path(__file__).parent / "lightning_multigpu_workers.yaml"
    with neps_config_file.open("r", encoding="utf-8") as fh:
        neps_config = yaml.load(fh, Loader=yaml.SafeLoader)
    max_eval_total = neps_config[MAX_EVALUATIONS_TOTAL]
    try:
        results, pending_configs = neps.status(base_dir)
        num_results = len(results)
    except FileNotFoundError:
        num_results = 0
    if num_results >= max_eval_total:
        print_with_rank(f"Reached max evaluations: {num_results}/{max_eval_total}")
        print_with_rank(f"Result directory: {base_dir}")
        sys.exit(1)  # exitcode 1 tells bash to not start another run

    if is_main_process():
        print_with_rank(f"Main NEPS worker pid={os.getpid()}")
        run_neps_main_debug(base_dir, neps_config_file)
        print_with_rank(f"Result directory: {base_dir}")
        sys.exit(0)

    print_with_rank(f"Non-NEPS worker pid={os.getpid()}")
    while True:
        # at this point there is no process group so we cannot dist.barrier. use filesystem instead.
        time.sleep(2)
        info_file = base_dir / f"rank0_info.json"
        if not info_file.is_file():
            continue

        # assert info_file.is_file(), f"Info file not found: {info_file} on rank {rank}/{world_size}"
        with info_file.open("r", encoding="utf-8") as fh:
            rank0_info = json.load(fh)
        train_config, checkpoint_dir, do_resume = rank0_info
        print_with_rank(f"========================================== Starting in {checkpoint_dir}")
        train_loss = train_model_debug(train_config, checkpoint_dir, do_resume)
        # to kill or not to kill this rank 1+ process?
        # ending it here will freeze the next run. so it seems lightning can reuse the
        # existing process group when we create a new trainer with same number of devices.
        # however, probably the safest options is to kill the process after one hpo config run
        # 1) by breaking here and letting the worker end, and and
        # 2) setting max_evaluations_per_run=1 in neps.run
        sys.exit(0)


def run_neps_main_debug(neps_root_dir, neps_config_file):
    neps.run(
        run_pipeline=run_pipeline_main_debug,
        run_args=Path(neps_config_file).as_posix(),
        root_directory=Path(neps_root_dir).as_posix(),
    )


def run_pipeline_main_debug(
    pipeline_directory: Path, previous_pipeline_directory: Path, **neps_config
) -> dict | float:
    # setup neps run
    if previous_pipeline_directory:
        init_dir = get_initial_directory(previous_pipeline_directory)
    else:
        init_dir = get_initial_directory(pipeline_directory)
    output_dir = f"{pipeline_directory}/run"
    checkpoint_dir = f"{init_dir}/ckpt"
    max_epochs_for_scheduler = neps_config.pop("max_epochs_for_scheduler")
    epochs = round(neps_config.pop("epochs"))
    pipeline_config = pipeline_directory.name
    previous_config = previous_pipeline_directory.name if previous_pipeline_directory else "none"
    # output_dir is the new config. ckpt_dir is either new or previous if that exists.
    ckpt_config_name = Path(checkpoint_dir).parts[-2]
    print_with_rank(
        f"E{epochs:2d} {pipeline_config:15s} ckpt: {ckpt_config_name:15s}  "
        f"previous: {previous_config:15s} "
    )
    lr = neps_config.pop("learning_rate")
    wd = neps_config.pop("weight_decay")
    train_config = {"lr": lr, "wd": wd}
    config_name = pipeline_directory.name  # e.g. config_34_1
    # pre-download dataset to avoid multiple processes downloading the same file and breaking
    _train_dataset = MNIST(os.getcwd(), download=True, transform=ToTensor())
    _val_dataset = MNIST(os.getcwd(), download=True, train=False, transform=ToTensor())
    print(f"========================================== MAIN starting in {output_dir}")
    rank0_info_file = pipeline_directory.parent.parent / f"rank0_info.json"
    assert not rank0_info_file.is_file(), f"Info file already exists: {rank0_info_file}"
    rank0_info = (train_config, output_dir, checkpoint_dir)
    with rank0_info_file.open("w", encoding="utf-8") as fh:
        json.dump(rank0_info, fh)
    train_loss = train_model_debug(train_config, output_dir, checkpoint_dir)
    os.unlink(rank0_info_file)
    time.sleep(1)
    return train_loss.item()


def train_model_debug(train_config, output_dir, checkpoint_dir):
    # in usual setting, output_dir and checkpoint_dir would be used to resume training
    # when using epochs as fidelity.
    autoencoder = LitAutoEncoder(lr=train_config["lr"], wd=train_config["wd"])
    train_dataset = MNIST(os.getcwd(), download=False, transform=ToTensor())
    val_dataset = MNIST(os.getcwd(), download=False, train=False, transform=ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)
    trainer = L.Trainer(
        limit_train_batches=100, max_epochs=1, devices=NUM_GPUS, accelerator="gpu", strategy="ddp"
    )
    trainer.fit(model=autoencoder, train_dataloaders=train_loader, val_dataloaders=val_loader)
    train_loss = trainer.callback_metrics.get("train_loss")
    val_loss = trainer.callback_metrics.get("val_loss")
    print_with_rank(f"train loss: {train_loss.item():.3f} val loss: {val_loss.item():.3f}")
    del trainer
    return train_loss


class LitAutoEncoder(L.LightningModule):
    def __init__(self, lr=1e-3, wd=1e-5):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
        self.decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))
        self.lr = lr
        self.wd = wd

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard (if installed) by default
        self.log("val_loss", loss, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        return optimizer


def get_process_group_key():
    """
    We need some sort of identifier to figure out the ID of this neps worker. Here: SLURM_JOB_ID
    If you are not on slurm you need to find another way to get a unique identifier.
    """
    job_id = None
    possible_job_id_keys = ["SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "MANUAL_JOB_ID"]
    for job_id_key in possible_job_id_keys:
        job_id = os.environ.get(job_id_key)
        if job_id is not None:
            break
    assert job_id is not None, f"No job id found after checking {possible_job_id_keys}"
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task_id is None:
        task_id = "null"
    process_group_key = f"{job_id}_{task_id}"
    return process_group_key


# ------------------------ copy paste distributed utils


def print_with_rank(*args, **kwargs):
    rank = get_rank()
    world_size = get_world_size()
    print(f"Rank {rank:>2d}/{world_size}:", *args, **kwargs)


def get_world_size() -> int:
    if is_slurm_sbatch():
        return int(os.environ["SLURM_NTASKS"])
    return int(os.environ.get("WORLD_SIZE", 1))


def get_rank() -> int:
    """
    In some cases LOCAL_RANK is set, but RANK is unset. Use LOCAL_RANK in that case.
    RANK: global rank of the process in the distributed setting, across all nodes.
    LOCAL_RANK: rank on this machine / node.

    in slurm sbatch scripts, we need to use the slurm env variables instead.
    """
    if is_slurm_sbatch():
        return int(os.environ["SLURM_PROCID"])

    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
    else:
        rank = int(os.environ.get("LOCAL_RANK", 0))
    return rank


def is_main_process():
    rank, _world_size = get_world_info()
    return rank == 0


def get_world_info() -> tuple[int, int]:
    return get_rank(), get_world_size()


def is_slurm_sbatch():
    slurm_job_name = os.environ.get("SLURM_JOB_NAME")
    if slurm_job_name is None:
        # not in slurm job
        return False
    if slurm_job_name == "bash":
        # in foreground slurm job
        return False
    # in background slurm job
    return True


if __name__ == "__main__":
    main()
