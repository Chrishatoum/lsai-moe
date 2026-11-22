#!/bin/bash
#SBATCH --job-name=a2a
#SBATCH --account=lsaiews26
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=00:15:00
#SBATCH --output=results/a2a-%j.out

export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n1)
export MASTER_PORT=29500
export NCCL_DEBUG=INFO

for n in 1 2 4; do
  srun -N$n --ntasks-per-node=4 --environment=pytorch python benchmarks/nccl/a2a.py
done
