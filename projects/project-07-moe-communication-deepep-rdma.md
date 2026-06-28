# Project 07: MoE Communication, DeepEP And RDMA Design

Recommended after: Module 16  
Estimated time: 25-45 小时

## Goal

为 MoE expert parallel 设计通信路径。重点不是复刻 DeepEP，而是理解 token routing、dispatch/combine、NVLink、RDMA、NCCL、buffer layout 和 expert GEMM 之间的关系。

## Work Modes

如果有多 GPU/多节点环境：

- 跑 NCCL tests 或最小 collective benchmark。
- 记录 `nvidia-smi topo -m`。
- 测 all-reduce、all-gather、all-to-all 或 P2P 的基础表现。
- 在满足当前 DeepEP README 要求（例如 SM90 级 GPU、匹配 CUDA/PyTorch/NCCL、RDMA 网络用于跨节点）时尝试 DeepEP examples；否则只做读码和实验计划。

如果没有环境：

- 做拓扑案例分析。
- 读 DeepEP README 和 examples。
- 做通信量预算和数据流设计。
- 写一份可实测实验计划。

## Required Deliverables

- GPU/NIC topology analysis。
- NCCL collective shape table。
- MoE dispatch/combine dataflow diagram。
- Token routing imbalance analysis。
- Prefill high-throughput vs decode low-latency comparison。
- RDMA caveat list: registration cache, NUMA, IOMMU, peer memory, buffer lifetime。
- Integration note for vLLM/SGLang。

## Communication Budget Template

```markdown
Model:
Experts:
Top-k:
Tokens:
Hidden size:
DType:
Dispatch bytes:
Combine bytes:
Metadata:
Imbalance assumption:
Overlap opportunities:
Main bottleneck hypothesis:
```

## Expert Bar

你通过本项目的标准是：能说明为什么 MoE 的瓶颈不只在 expert GEMM，还在 token 如何跨 GPU 到达 expert，以及这些路径如何被硬件拓扑限制。
