# LSAIE Track A — MoE expert granularity vs. network

How fine can MoE experts get before expert-parallel communication dominates throughput on GH200 (Alps)?

- `theory/` roofline prediction of the communication cliff
- `bench/` benchmark harness for training, serving and network (see bench/README.md)
- `env/` container definitions
- `results/` raw logs
- `docs/explainer.md` plain-language explanation of the project
