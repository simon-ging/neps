#!/bin/bash -l
#SBATCH -p XXXXXXXX
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --gres gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=40G
#SBATCH -a 1-1%1
#SBATCH -o $HOME/slurm_logs/%x.%N.%A.%a.log
#SBATCH -J nepstest

source "$HOME/start_conda_on_slurm.sh"
which conda
conda --version
conda activate nepstest
which python
python -V
pwd
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

echo "world size  SLURM_NTASKS=      $SLURM_NTASKS"
echo "global rank SLURM_PROCID=      $SLURM_PROCID"
echo "local rank  SLURM_LOCALID=     $SLURM_LOCALID"
echo "node_rank   SLURM_NODEID=      $SLURM_NODEID"
echo "cpus        SLURM_CPUS_ON_NODE=$SLURM_CPUS_ON_NODE"

while true; do
    echo -e "\n ----------- new config\n"
    srun python -u lightning_multigpu_workers.py
    RETCODE=$?
    if [ $RETCODE -ne 0 ]; then
        echo -e "\n---------- exiting with code $RETCODE\n"
        break
    fi
    sleep 5
    echo
done
