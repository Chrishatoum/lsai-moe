#!/bin/bash
#SBATCH --account=lsaiews26
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=00:40:00
#SBATCH --output=results/logs/serve-%j.out
set -e
cd ~/lsai-moe
source env/containers.sh
SMODEL=${SMODEL:-Qwen/Qwen3-30B-A3B}; LEVER=${LEVER:-baseline}; GPUS=${GPUS:-4}
IN=${IN:-1024}; OUT=${OUT:-512}; N=${N:-512}; RATE=${RATE:-inf}

FLAGS=(--tensor-parallel-size $GPUS --max-model-len $((IN + OUT + 128)))
EAGER=1; PREFIX=0
for l in ${LEVER//,/ }; do
  case $l in
    baseline) ;;
    cudagraph) EAGER=0 ;;
    ep) FLAGS+=(--enable-expert-parallel) ;;
    fp8kv) FLAGS+=(--kv-cache-dtype fp8) ;;
    prefix) PREFIX=1 ;;
    *) echo "unknown lever $l"; exit 1 ;;
  esac
done
[ $EAGER = 1 ] && FLAGS+=(--enforce-eager)
[ $PREFIX = 0 ] && FLAGS+=(--no-enable-prefix-caching)

echo "BENCH kind=serve model=$SMODEL lever=$LEVER gpus=$GPUS in=$IN out=$OUT n=$N rate=$RATE"

srun --environment=vllm bash -c "
  vllm serve $SMODEL ${FLAGS[*]} --port 8000 > results/logs/vllm-server-$SLURM_JOB_ID.log 2>&1 &
  until curl -sf localhost:8000/health >/dev/null; do sleep 10; done
  vllm bench serve --model $SMODEL --port 8000 --dataset-name random \
    --random-input-len $IN --random-output-len $OUT --num-prompts $N --request-rate $RATE \
    --percentile-metrics ttft,tpot,itl --metric-percentiles 50,90,99 --ignore-eos
  kill %1"
