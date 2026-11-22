# MoE expert granularity vs. network on GH200

ETH Zürich, Large-Scale AI Engineering, Autumn 2026 — Track A (training and serving throughput).

**Question:** how fine can Mixture-of-Experts experts get before expert-parallel communication dominates training throughput?

Fine-grained experts (many small experts, more of them per token) improve model quality at fixed compute, but every token is copied to each of its experts. Splitting experts N times finer keeps compute per token constant while multiplying network traffic by N. Inside a node GPUs talk over NVLink; across nodes over Slingshot, which is several times slower. At some expert size the GPUs start waiting on the network.

Following the DeepSeek-V4 analysis, each token-expert pair costs `6·h·d` FLOPs and `3·h` bytes, so communication stays hidden while

```
expert width d  ≥  d* = C / (2·B)
```

with `C` the achieved compute per GPU and `B` the all-to-all bandwidth per GPU. We measure `B`, predict `d*`, then check the prediction against training runs and test which optimisations move it.

## Setup

- Cluster: CSCS Clariden (Alps), GH200, 4 GPUs per node
- Training: Megatron-LM `core_v0.14.0` in NGC PyTorch 25.06
- Serving: vLLM in NGC vLLM 26.08

Models (hidden 2048, 16 layers, sequence 4096). Compute per token and total expert parameters are identical across `g*`; only granularity changes.

| Model | Experts | Experts per token | Expert width |
|---|---|---|---|
| dense | – | – | FFN 8192 |
| g1 | 8 | 1 | 8192 |
| g2 | 16 | 2 | 4096 |
| g4 | 32 | 4 | 2048 |
| g8 | 64 | 8 | 1024 |
| g16 | 128 | 16 | 512 |

## Results so far

All-to-all bandwidth per GPU (256 MB messages):

| Nodes | GPUs | Bandwidth |
|---|---|---|
| 1 | 4 | 262 GB/s |
| 2 | 8 | 40 GB/s |
| 4, 8 | 16, 32 | pending |

Training throughput, baseline (no optimisations), tokens/s/GPU:

| Model | 1 node | 2 nodes | 4 nodes | 8 nodes |
|---|---|---|---|---|
| dense | 41,374 | 39,453 | 27,925 | 34,149 |
| g2 | 21,080 | 23,024 | 21,351 | – |
| g4 | 15,843 | 16,379 | 15,122 | 11,519 |
| g8 | 10,504 | 12,585 | 8,844 | 6,618 |
| g16 | 6,122 | 7,237 | 4,225 | 3,995 |

Finer experts are slower even on one node (no grouped GEMM: experts run sequentially) and lose more across nodes. A grouped-GEMM sweep separating the two effects is in progress.

Serving, Qwen3-30B-A3B, 1 node, TP=4, baseline: 1,922 output tokens/s/GPU, median TTFT 3.3 s, median TPOT 50 ms.

Full tables: `results/train.csv`, `results/serve.csv`. Status page: `docs/results.html`.

## Repository

| Path | Contents |
|---|---|
| `env/` | container definitions and `setup.sh` (imports containers, clones Megatron-LM) |
| `bench/` | benchmark harness, see [`bench/README.md`](bench/README.md) |
| `bench/train/` | model definitions, optimisation flags, Megatron job script |
| `bench/serve/` | vLLM job script |
| `bench/nccl/` | all-to-all bandwidth benchmark |
| `theory/roofline.py` | predicted communication cliff from measured bandwidth |
| `results/` | raw job logs and parsed CSVs |
| `docs/` | plain-language explainer and results page |

## Reproduce

```bash
./env/setup.sh                                          # once: containers + Megatron-LM

for n in 1 2 4 8; do sbatch -N $n bench/nccl/a2a.sbatch; done   # network bandwidth
python3 theory/roofline.py 300                          # predicted cliff at 300 TFLOP/s per GPU

bench/sweep.sh baseline 1 2 4 8                         # all models x node counts
bench/submit.sh train g16 fp8 4                         # one model, one optimisation, 4 nodes
bench/submit.sh serve Qwen/Qwen3-30B-A3B cudagraph      # serving with one optimisation

python3 bench/parse.py                                  # logs -> results/train.csv, results/serve.csv
```

Training optimisations: `grouped_gemm`, `fusion`, `overlap_a2a`, `fp8`, `distopt`, `recompute`, `deepep`, `grouplimit` (comma-separated to combine). Serving: `cudagraph`, `ep`, `fp8kv`, `prefix`.
