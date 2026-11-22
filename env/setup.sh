#!/bin/bash
set -e
cd "$(dirname "$0")/.."
source env/containers.sh
mkdir -p ~/.edf "$HF_HOME" results/logs
for t in pytorch vllm; do
  sed -e "s|SCRATCH_PATH|$SQ|g" -e "s|REPO_PATH|$PWD|g" -e "s|HOME_PATH|$HOME|g" env/$t.toml > ~/.edf/$t.toml
done

run() { srun --account=lsaiews26 --partition=debug -N1 -t 00:30:00 "$@"; }

run nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | sed 's/^/driver: /'
[ -f "$SQ/pytorch-25.06.sqsh" ] || run enroot import -o "$SQ/pytorch-25.06.sqsh" "docker://$PYTORCH_IMG"
[ -f "$SQ/vllm-26.08.sqsh" ] || run enroot import -o "$SQ/vllm-26.08.sqsh" "docker://$VLLM_IMG"
[ -d "$MEGATRON" ] || git clone --depth 1 --branch "$MEGATRON_REF" https://github.com/NVIDIA/Megatron-LM.git "$MEGATRON" \
  || git clone --depth 1 https://github.com/NVIDIA/Megatron-LM.git "$MEGATRON"
git -C "$MEGATRON" log -1 --format='megatron: %h %s'
