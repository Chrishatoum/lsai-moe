#!/bin/bash
#SBATCH --job-name=a2a
#SBATCH --account=lsaiews26
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=00:10:00
#SBATCH --output=results/logs/a2a-%j.out
# run once per node count: for n in 1 2 4 8; do sbatch -N $n bench/nccl/a2a.sbatch; done
export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n1)
export MASTER_PORT=29500
export NCCL_DEBUG=INFO
srun --environment=pytorch python bench/nccl/a2a.py
