#!/bin/bash
#SBATCH --account=lsaiews26
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=00:20:00
#SBATCH --output=results/logs/train-%j.out
set -e
cd "$SLURM_SUBMIT_DIR"
source env/containers.sh
MODEL=${MODEL:-g4}; LEVER=${LEVER:-baseline}; EP=${EP:-$SLURM_NTASKS}
source bench/train/models.sh
source bench/train/levers.sh
[ "$MODEL" = dense ] && EP=1
MBS=1; GBS=$((SLURM_NTASKS * 4))

echo "BENCH kind=train model=$MODEL lever=$LEVER nodes=$SLURM_NNODES gpus=$SLURM_NTASKS ep=$EP seq=$SEQ gbs=$GBS layers=$LAYERS"

ARGS=(
  "${MODEL_ARGS[@]}" "${LEVER_ARGS[@]}"
  --expert-model-parallel-size $EP --expert-tensor-parallel-size 1
  --micro-batch-size $MBS --global-batch-size $GBS --train-iters 15
  --lr 1e-4 --min-lr 1e-5 --lr-decay-style cosine --weight-decay 0.1 --clip-grad 1.0
  --bf16 --transformer-impl transformer_engine
  --mock-data --tokenizer-type NullTokenizer --vocab-size 32000
  --eval-iters 0 --log-interval 1 --log-throughput
)

export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n1) MASTER_PORT=29500
export CUDA_DEVICE_MAX_CONNECTIONS=$CDMC
srun --environment=pytorch bash -c "
  export RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID WORLD_SIZE=\$SLURM_NTASKS PYTHONPATH=$MEGATRON
  python $MEGATRON/pretrain_gpt.py ${ARGS[*]}"
