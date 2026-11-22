#!/bin/bash
# bench/submit.sh train MODEL LEVER NODES [EP]
# bench/submit.sh serve HF_MODEL LEVER [GPUS]
cd "$(dirname "$0")/.."
case $1 in
  train) MODEL=$2 LEVER=$3 EP=$5 sbatch -N ${4:-1} --job-name=t-$2-$3-n${4:-1} --export=ALL bench/train/train.sbatch ;;
  serve) SMODEL=$2 LEVER=$3 GPUS=${4:-4} sbatch --job-name=s-${2##*/}-$3 --export=ALL bench/serve/serve.sbatch ;;
  *) sed -n 2,3p "$0" ;;
esac
