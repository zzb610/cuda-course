# Project 08: Triton, TileLang, CUDA And CUTLASS Comparison

Recommended after: Module 19  
Estimated time: 25-45 小时

## Goal

同一个算子用多种实现路线比较。目标不是证明某个工具永远最好，而是训练你做技术选型：开发速度、性能、调试、部署、框架接入和硬件能力之间如何取舍。

## Candidate Operators

- `fused_bias_gelu`
- `rmsnorm_forward`
- simple matmul tile
- top-k softmax simplified variant
- quantize/dequantize per token

## Required Implementations

至少完成两种，鼓励三种：

- PyTorch baseline。
- CUDA C++ custom op。
- Triton kernel。
- TileLang case study or implementation。
- CUTLASS/CuTe or DeepGEMM case study if GEMM-like。

## Required Deliverables

- Same input shape set for all implementations。
- Correctness reference and tolerance。
- Benchmark method。
- Code complexity comparison。
- Profiler observations。
- Build/deploy friction notes。
- Framework integration plan。
- Final decision matrix。

## Decision Matrix

| Route | Performance | Dev speed | Debug cost | Portability | Framework integration | Best use |
|---|---|---|---|---|---|---|
| CUDA C++ | | | | | | |
| Triton | | | | | | |
| TileLang | | | | | | |
| CUTLASS/CuTe | | | | | | |

## Expert Bar

最终结论必须可辩护：不是“我喜欢 Triton/CUDA”，而是“在这些 shape、GPU、部署约束、维护成本下，我选择这个实现，并保留这些 fallback”。
