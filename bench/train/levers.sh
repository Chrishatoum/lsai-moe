# LEVER is a comma-separated list, e.g. "grouped_gemm,fusion,fp8"; "baseline" adds nothing.
LEVER_ARGS=(--moe-token-dispatcher-type alltoall)
CDMC=1
for l in ${LEVER//,/ }; do
  case $l in
    baseline) ;;
    grouped_gemm) LEVER_ARGS+=(--moe-grouped-gemm) ;;
    fusion) LEVER_ARGS+=(--moe-permute-fusion --moe-router-fusion) ;;
    overlap_a2a) LEVER_ARGS+=(--overlap-moe-expert-parallel-comm --delay-wgrad-compute); CDMC=32 ;;
    fp8) LEVER_ARGS+=(--fp8-format e4m3 --fp8-recipe blockwise --moe-router-padding-for-fp8) ;;
    distopt) LEVER_ARGS+=(--use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather) ;;
    recompute) LEVER_ARGS+=(--recompute-granularity selective) ;;
    deepep) LEVER_ARGS+=(--moe-token-dispatcher-type flex --moe-enable-deepep) ;;
    grouplimit) LEVER_ARGS+=(--moe-router-num-groups ${NGROUPS:-$SLURM_NNODES} --moe-router-group-topk ${GTOPK:-1}) ;;
    *) echo "unknown lever $l"; exit 1 ;;
  esac
done
