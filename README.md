# LSAIE Track A — MoE expert granularity vs. network

How fine can MoE experts get before expert-parallel communication dominates throughput on GH200 (Alps)?

- `theory/` roofline prediction of the communication cliff
- `benchmarks/nccl/` measured all-to-all bandwidth, intra- vs inter-node
- `env/` container definitions
- `results/` raw logs
