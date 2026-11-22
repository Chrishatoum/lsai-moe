# Project explainer

## The course task (Track A)

Make AI models run as fast as possible on the GH200 GPUs of Clariden, and show which tricks make them faster.

- **Training** = teaching a model. We build our own models from random weights, train them for 15 steps on fake data and measure speed only (tokens per second per GPU). Tool: Megatron-LM.
- **Serving** = using a finished model to answer requests. We use released models (Qwen3, Mixtral, gpt-oss) and measure generated tokens per second per GPU and latency. Tool: vLLM.

For both: report a baseline (no tricks), each trick on its own, and the best combination.

## Hardware: nodes and GPUs

A **node** is one server with **4 GPUs**.

```
NODE 1                              NODE 2
GPU 0   GPU 1   GPU 2   GPU 3       GPU 4   GPU 5   GPU 6   GPU 7
   └──── NVLink (fast) ────┘           └──── NVLink (fast) ────┘
                 └──────── Slingshot (slow) ────────┘
```

- Inside a node, GPUs talk over **NVLink**: very fast.
- Between nodes, they talk over **Slingshot**: about 6–20× slower.

## How a Mixture-of-Experts (MoE) model works

```
MODEL
├── Layer 1
│   ├── Attention      (tokens look at each other)
│   └── MoE block
│       ├── Router     (picks which experts each token uses)
│       └── Experts    (small neural networks, blocks of learned numbers)
├── Layer 2 ... Layer 16 (each with its own router and experts)
```

- A **token** is a piece of text, stored as **2048 numbers**. This never changes.
- An **expert** is a small neural network inside the model. It takes a token (2048 numbers), expands it to its internal width, processes it, and shrinks it back to 2048.
- The **router** picks which experts each token goes to. **Experts per token** (top-k) is how many it picks.
- The token is **copied** to each chosen expert. Each expert processes the whole token at the same time as the others, and the results are added together.

Experts don't fit on one GPU, so they are **spread evenly across GPUs** (expert parallelism). When a token's expert lives on another GPU, the token has to be **sent** there and the result **sent back**.

## Our models: g1 to g16

The number after **g** says: each token goes to N experts, each expert is N times smaller, and there are N times as many experts.

| Model | Experts per token | Expert width | Total experts | Work per token |
|---|---|---|---|---|
| g1 | 1 | 8192 | 8 | 8192 |
| g2 | 2 | 4096 | 16 | 8192 |
| g4 | 4 | 2048 | 32 | 8192 |
| g8 | 8 | 1024 | 64 | 8192 |
| g16 | 16 | 512 | 128 | 8192 |
| dense | normal model, no experts | 8192 | – | 8192 |

**Why does g4 send each token to 4 experts?** Each g4 expert is 4× smaller than a g1 expert. With only 1, the token would get a quarter of the processing: a cheaper, weaker model, and an unfair comparison. With 4, every model does the **same work per token**, so the only difference between them is how finely the work is split.

**Why fine-grained at all?** Labs like DeepSeek and Qwen found many small experts give better model quality for the same compute. That is why the speed question matters.

## Where the experts live

Experts per GPU = total experts ÷ number of GPUs. Expert size is part of the model and never changes with the number of nodes; only where the experts live changes.

| Model | 1 node (4 GPUs) | 2 nodes (8 GPUs) | 4 nodes (16 GPUs) | 8 nodes (32 GPUs) |
|---|---|---|---|---|
| g1 (8) | 2 per GPU | 1 per GPU | – | – |
| g2 (16) | 4 | 2 | 1 | – |
| g4 (32) | 8 | 4 | 2 | 1 |
| g16 (128) | 32 | 16 | 8 | 4 |

A dash means the model has fewer experts than GPUs, so it can't be spread that far.

The more nodes, the larger the share of experts in other nodes, so more copies cross slow Slingshot:

| Nodes | Share of other experts in other nodes |
|---|---|
| 1 | 0% (all NVLink) |
| 2 | ~50% |
| 4 | ~75% |
| 8 | ~88% |

## Example: one token, g4 on 2 nodes

g4: 32 experts, 4 per GPU, each token goes to 4.

```
GPU 0: experts 0–3      GPU 4: experts 16–19
GPU 1: experts 4–7      GPU 5: experts 20–23
GPU 2: experts 8–11     GPU 6: experts 24–27
GPU 3: experts 12–15    GPU 7: experts 28–31
```

Token "cat" is on GPU 0.

1. **Router** picks experts 2, 9, 18, 30.
2. **Send** a copy of "cat" to each:

| Expert | Lives on | Path | Speed |
|---|---|---|---|
| 2 | GPU 0 | stays | instant |
| 9 | GPU 2 | NVLink | fast |
| 18 | GPU 4 | Slingshot | slow |
| 30 | GPU 7 | Slingshot | slow |

3. **Compute**: each expert processes its copy.
4. **Send back** the 4 results.
5. **Combine**: add them up; "cat" moves to the next layer.

Every GPU does this for thousands of tokens at once, in every layer.

## Why small experts are slow across nodes

- Every copy sent is the full token (2048 numbers), whatever the expert size.
- **gN sends N copies per token**, but each expert does N times less work.
- Total work stays the same; network traffic grows N times.

GPUs can send and compute at the same time. While an expert works on one token, the next is already on its way.

- **Big expert**: long compute, so the next token arrives in time. Sending is hidden, no waiting.
- **Tiny expert on several nodes**: short compute, many slow copies. The expert finishes and waits for Slingshot. Speed drops.
- **Tiny expert on 1 node**: NVLink is fast enough that even tiny experts don't wait.

Restaurant version: chefs are experts, kitchens are GPUs, buildings are nodes. Trays move instantly inside a building and by slow van between buildings. One big chef cooking 16 minutes never waits for the van. Sixteen small chefs cooking 1 minute each, spread over several buildings, spend most of their time waiting for vans.

## The prediction

From the DeepSeek-V4 paper: per token and expert, compute is 6·h·d FLOPs and traffic is 3·h bytes (FP8 send, BF16 return). Sending stays hidden when

```
GPU compute speed / network speed  ≤  2 · d
```

so the smallest expert width that still hides the network is `d* = C / (2·B)`.

Rough numbers (C ≈ 400 TFLOP/s achieved):

| Where | Bandwidth B per GPU | d* |
|---|---|---|
| Inside one node | ~600 GB/s | ~330 |
| Across nodes | ~25 GB/s | ~8000 |

Inside a node all our models should be fine. Across nodes even g1 is near the limit and small experts should slow down a lot. We measure the real B with the a2a benchmark and recompute d*.

## What we test

1. **Network**: real NVLink and Slingshot bandwidth (`sbatch bench/nccl/a2a.sbatch`).
2. **Predict** where the slowdown starts, from that bandwidth.
3. **Baseline sweep**: every model on every node count, no tricks.

| | 1 node | 2 nodes | 4 nodes | 8 nodes |
|---|---|---|---|---|
| g1 | ✓ | ✓ | – | – |
| g2 | ✓ | ✓ | ✓ | – |
| g4 | ✓ | ✓ | ✓ | ✓ |
| g8 | ✓ | ✓ | ✓ | ✓ |
| g16 | ✓ | ✓ | ✓ | ✓ |

```bash
for g in g1 g2 g4 g8 g16; do for n in 1 2 4 8; do bench/submit.sh train $g baseline $n; done; done
```

Expected: flat on 1 node; across nodes, speed drops as experts get smaller. We look for where it turns from fast to slow and compare with the prediction.

4. **Tricks**, one at a time, on a slow case (e.g. g16 on 4 nodes), then combined:

| Trick | What it does |
|---|---|
| `grouped_gemm` | all experts on a GPU compute in one batch instead of one by one |
| `fusion` | merges many small GPU operations into fewer, bigger ones |
| `fp8` | 8-bit numbers: faster maths, half the data to send |
| `overlap_a2a` | sends the next tokens while computing the current ones |
| `grouplimit` | each token may only use experts on a limited number of nodes |
| `deepep` | faster token-sending library from DeepSeek (likely only inside a node on Slingshot) |
| `distopt` | spreads optimizer memory across GPUs, overlaps gradient sync |
| `recompute` | recomputes some intermediate results to save memory |

```bash
for t in grouped_gemm fusion fp8 overlap_a2a grouplimit; do bench/submit.sh train g16 $t 4; done
bench/submit.sh train g16 grouped_gemm,fusion,fp8,overlap_a2a 4
```

5. **Serving**, same idea with a released MoE:

| Trick | What it does |
|---|---|
| `baseline` | eager mode, no prefix caching |
| `cudagraph` | records GPU operations once and replays them |
| `ep` | whole experts on different GPUs instead of splitting each expert across GPUs |
| `fp8kv` | stores the conversation memory (KV cache) in 8-bit |
| `prefix` | reuses work when requests share the same start |
| FP8/MXFP4 model | a released lower-precision checkpoint, e.g. `Qwen/Qwen3-30B-A3B-FP8` |

```bash
for t in baseline cudagraph ep fp8kv; do bench/submit.sh serve Qwen/Qwen3-30B-A3B $t; done
```

All results land in `results/train.csv` and `results/serve.csv` after `python bench/parse.py`.

## The expected result

A chart of speed against expert size: the 1-node line stays flat, the multi-node lines fall off a cliff, and lines with tricks fall later. The report says where we predicted the cliff, where we measured it, and how much each trick moved it.
