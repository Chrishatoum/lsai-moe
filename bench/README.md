# Benchmarks

Every run writes `results/logs/<kind>-<jobid>.out` with a `BENCH ...` header. `python bench/parse.py` collects them into `results/train.csv` and `results/serve.csv`.

## Training (Megatron-LM, mock data, 15 iters, metric = mean of last 5)

```bash
bench/submit.sh train MODEL LEVER NODES [EP]
bench/submit.sh train g4 baseline 1
bench/submit.sh train g4 grouped_gemm,fusion,fp8 2
```

| MODEL | experts | top-k | expert width d |
|---|---|---|---|
| dense | - | - | ffn 8192 (same active compute) |
| g1 | 8 | 1 | 8192 |
| g2 | 16 | 2 | 4096 |
| g4 | 32 | 4 | 2048 |
| g8 | 64 | 8 | 1024 |
| g16 | 128 | 16 | 512 |

h = 2048, 16 layers, seq 4096. Active FFN compute and total expert parameters are identical across g*, so only granularity changes. EP must divide the expert count (g1 runs at most EP=8). `LAYERS=4` gives a smaller model for single-GPU (EP=1, no-communication) controls.

| LEVER | What it tests |
|---|---|
| baseline | NCCL all-to-all, sequential experts, BF16 |
| grouped_gemm | one batched GEMM for all local experts |
| fusion | fused permute/unpermute and router kernels |
| overlap_a2a | hide EP all-to-all behind compute (sets CUDA_DEVICE_MAX_CONNECTIONS=32) |
| fp8 | FP8 blockwise training via Transformer Engine |
| distopt | distributed optimizer, grad-reduce and param-gather overlap |
| recompute | selective activation recomputation |
| deepep | DeepEP dispatcher (needs DeepEP installed; inter-node unlikely on Slingshot) |
| grouplimit | each token routed to at most GTOPK of NGROUPS node groups |

Combine levers with commas. Report each lever alone against baseline, then the best combination.

Whole grid in one go: `bench/sweep.sh baseline 1 2 4 8` (skips combinations with fewer experts than GPUs).

## Serving (vLLM, one node, random prompts, `vllm bench serve`)

```bash
bench/submit.sh serve HF_MODEL LEVER [GPUS]
bench/submit.sh serve Qwen/Qwen3-30B-A3B baseline
bench/submit.sh serve Qwen/Qwen3-30B-A3B cudagraph,ep
```

| LEVER | What it tests |
|---|---|
| baseline | TP across GPUs, eager mode, no prefix caching |
| cudagraph | CUDA graphs on |
| ep | expert parallelism instead of TP for MoE layers |
| fp8kv | FP8 KV cache |
| prefix | prefix caching on |

Workload via env: `IN`, `OUT`, `N`, `RATE` (default 1024/512/512/inf). FP8 or MXFP4 weights by choosing a released checkpoint, e.g. `Qwen/Qwen3-30B-A3B-FP8`, `openai/gpt-oss-120b`.

## Network

`sbatch bench/nccl/a2a.sbatch` measures all-to-all bandwidth on 1/2/4 nodes.
