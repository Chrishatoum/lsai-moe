#!/bin/bash
set -e
img=/iopsstor/scratch/cscs/$USER/pytorch-25.06.sqsh
mkdir -p ~/.edf
cp env/pytorch.toml ~/.edf/pytorch.toml
[ -f "$img" ] || srun --account=lsaiews26 --partition=debug -N1 -t 00:30:00 \
  enroot import -o "$img" docker://nvcr.io#nvidia/pytorch:25.06-py3
