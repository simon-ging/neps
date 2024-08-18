"""
Tested environment: ubuntu, latest torch as of 2024-08, slurm cluster, 2xRTX2080Ti

Usage:
    conda create -n nepstest python=3.10 -y
    conda activate nepstest
    pip install torch torchvision lightning neps neural-pipeline-search
    # get a slurm job with 2 gpus (or change NUM_GPUS below)
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
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

import neps
from neps.utils.common import get_initial_directory
from neps.utils.run_args import MAX_EVALUATIONS_TOTAL
from packg.iotools import load_yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NUM_GPUS = 2


def ensure_neps_config(base_dir):
    # for debugging only. in a normal use case this file is created manually
    base_dir = Path(base_dir)
    yaml_content = """
    pipeline_space:
        max_epochs_for_scheduler: 9
        learning_rate:
            lower: 1e-5
            upper: 1e-3
            log: true
            default: 1e-4
        weight_decay:
            lower: 1e-5
            upper: 2e-1
            log: true
            default: 1e-2
        epochs:
            lower: 1
            upper: 9
            is_fidelity: true

    searcher:
        strategy: priorband
        eta: 3

    max_evaluations_total: 10
    max_evaluations_per_run: 1
    post_run_summary: true
    ignore_errors: false
    """
    yaml_file = base_dir / "example_neps_config.yaml"
    if not yaml_file.is_file() and is_main_process():
        os.makedirs(base_dir, exist_ok=True)
        Path(yaml_file).write_text(yaml_content, encoding="utf-8")
    return yaml_file


def main():
    process_group_key = get_process_group_key()
    print_with_rank(f"{process_group_key=}")
    base_dir = Path.home() / f".cache/neps_debug/{process_group_key}"

    if is_main_process():
        neps_config_file = ensure_neps_config(base_dir)
        neps_config = load_yaml(neps_config_file)
        max_eval_total = neps_config[MAX_EVALUATIONS_TOTAL]
        try:
            results, pending_configs = neps.status(base_dir)
            num_results = len(results)
        except FileNotFoundError:
            num_results = 0
        if num_results >= max_eval_total:
            print_with_rank(f"Reached max evaluations: {num_results}/{max_eval_total}")
            sys.exit(1)  # exitcode 1 tells bash to not start another run
        print_with_rank(f"Main NEPS worker pid={os.getpid()}")
        run_neps_main_debug(base_dir, neps_config_file)
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
        train_config, output_dir, checkpoint_dir = rank0_info
        print_with_rank(f"========================================== Starting in {output_dir}")
        train_loss = train_model_debug(train_config, output_dir, checkpoint_dir)
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
    print(f"Check root directory for the CSV results: {neps_root_dir}")


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
    print(f"========================================== MAIN starting in {output_dir}")
    # neps_debug/11754531_null/root/results/config_1_0
    rank0_info_file = pipeline_directory.parent.parent / f"rank0_info.json"
    assert not rank0_info_file.is_file(), f"Info file already exists: {rank0_info_file}"
    rank0_info = (train_config, output_dir, checkpoint_dir)
    with rank0_info_file.open("w", encoding="utf-8") as fh:
        json.dump(rank0_info, fh)
    train_loss = train_model_debug(train_config, output_dir, checkpoint_dir)
    os.unlink(rank0_info_file)
    time.sleep(1)
    return train_loss.item()

    # # so now we DO NOT have a process group yet because we haven't called trainer.fit
    # # at this point dist process group exists so we could use that to broadcast the config_num
    # rank0_info = (train_config, output_dir, checkpoint_dir)
    # dump_json(rank0_info, pipeline_directory / f"rank0_info_{config_num}.json")
    # config_num_tensor = torch.tensor(config_num, dtype=torch.int64).to(torch.device("cuda:0"))
    # dist.barrier()
    # dist.broadcast(config_num_tensor, src=0)
    #
    # # ------------------------ the training run
    # # return debug_run_training_multigpu(train_config, output_dir, checkpoint_dir, config_num)


def train_model_debug(train_config, output_dir, checkpoint_dir):
    autoencoder = LitAutoEncoder(lr=train_config["lr"], wd=train_config["wd"])
    train_dataset = MNIST(os.getcwd(), download=True, transform=ToTensor())
    val_dataset = MNIST(os.getcwd(), download=True, train=False, transform=ToTensor())
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


# define the LightningModule
class LitAutoEncoder(L.LightningModule):
    def __init__(self, lr=1e-3, wd=1e-5):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
        self.decoder =  nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))
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


def get_rank() -> int:
    if "RANK" in os.environ:  # global rank across all nodes
        rank = int(os.environ["RANK"])
    else:  # local rank on the node
        rank = int(os.environ.get("LOCAL_RANK", 0))
    return rank


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))


def is_main_process():
    rank, _world_size = get_world_info()
    return rank == 0


def get_world_info() -> tuple[int, int]:
    return get_rank(), get_world_size()


if __name__ == "__main__":
    main()
