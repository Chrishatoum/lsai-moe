# LSAIE Track A — MoE expert granularity vs. network

ETH LSAIE Autumn 2026 (Imanol Schlag). Team of 4: Chris (Chrishatoum), Darius (userdarius), Alexandros (Alexp34), Theo (0xtlf). This is Chris's personal repo until the team repo is accessible; history will be merged into it later.

## Course requirements (Track A)
- Highest training throughput (Megatron-LM, tokens/s/GPU over 5 steps after warm-up) and serving throughput (vLLM/SGLang, generated tokens/s/GPU at steady state + latency percentiles) on GH200. Larger models score better.
- Report unoptimised and optimised numbers, each lever in isolation, for both training and serving.
- Serving optimisations that need training must use released artefacts.
- Every script, config, data file and result is committed as work happens; the git history is the contribution log.
- Final deadline: Sun 20 Dec 2026 evening, `report.md` is the final report.

## Research question
How fine can MoE experts get before expert-parallel communication dominates throughput?
DeepSeek-V4 rule: each token-expert pair costs 6·h·d FLOPs and 3·h bytes (FP8 dispatch, BF16 combine), so communication hides when C/B ≤ 2d.
- Predict the cliff d* = C/2B per EP span (remote fraction 1 − 1/nodes), intra-node NVLink vs inter-node Slingshot (~25 GB/s/GPU).
- Measure: fixed-compute granularity sweep (k·d and E·d constant), EP = 4/8/16/32, plus no-communication controls to separate GEMM-efficiency loss from the comm cliff.
- Levers: grouped GEMM, permute fusion, `--overlap-moe-expert-parallel-comm --delay-wgrad-compute`, FP8 blockwise, DeepEP intra-node (inter-node DeepEP needs IBGDA, likely unavailable on Slingshot), group-limited routing.
- Serving: released MoEs (Qwen3-30B-A3B fine-grained, Mixtral coarse, gpt-oss MXFP4), TP vs EP intra/inter-node.

## Cluster
- Clariden (aarch64 GH200, 4 GPUs/node). User `course_00651`, SLURM account `lsaiews26`, partitions `normal` and `debug`.
- Job I/O on `/iopsstor/scratch/cscs/course_00651`; home has a 50 GB quota.
- Container: NGC PyTorch 25.06 squashfs on iopsstor, EDF `~/.edf/pytorch.toml` (from `env/pytorch.toml`) with the aws_ofi_nccl hook. Use `--environment=pytorch`.
- Never run compute on the login node; use `sbatch` or `srun --partition=debug`.

## Layout
- `env/` container definition and setup
- `benchmarks/nccl/` all-to-all bandwidth benchmark
- `theory/` roofline prediction
- `results/` raw logs

## Conventions
- Concise, minimal code; no excessive comments or docstrings.
- Small, frequent commits with clear messages.

## Status
- [x] Repo skeleton, all-to-all benchmark
- [ ] Container imported, a2a benchmark run on 1/2/4 nodes
- [ ] Roofline predictor using measured bandwidth
- [ ] Megatron MoE baseline and granularity sweep
