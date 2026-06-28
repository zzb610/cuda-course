# Weekly Checkpoints

## Module 00

提交环境记录，证明能构建并运行最小 CUDA 程序。

## Module 01

给定任意 `N` 和 `blockDim.x`，写出正确 grid size、kernel index 和 bounds check。

## Module 02

提交带错误检查、edge-case tests 和 Compute Sanitizer 记录的小程序。

## Module 03

解释 naive transpose 与 tiled transpose 的内存访问差异，并给出 profiling evidence。

## Module 04

实现任意长度 reduction，并比较 custom implementation 与 CUB baseline。

## Module 05

提交一份完整 profiling report，包含 hypothesis、baseline、measurement、change、result、conclusion。

## Module 06

为三个并行问题选择 Thrust、CUB、cuBLAS 或 custom kernel，并说明理由。

## Module 07

完成 custom vs library comparison，结论可以是保留库实现。

## Module 08

用 Nsight Systems 证明 stream pipeline 是否发生 overlap。

## Module 09

写 architecture caveat note，说明一个优化依赖哪些 GPU 能力。

## Module 10

提交 small CUDA library skeleton，包含 build、test、benchmark 和 API contract。

## Module 11

提交 capstone final memo 和可复现 artifact。

## Module 12

提交 Volta/Ampere/Hopper/Blackwell 架构对比图和 feature matrix，说明每代硬件能力如何改变 CUDA kernel 设计。

## Module 13

提交 naive/tiled/WMMA/CUTLASS 或 DeepGEMM 的 GEMM 分层报告，解释 Tensor Core、tile 层级、dtype、layout 和 epilogue。

## Module 14

提交 PTX/SASS 阅读笔记和 WGMMA/TMA 异步流水线图，说明 `mma.sync`、`wgmma.mma_async`、TMA、mbarrier 的位置。

## Module 15

提交现代 GEMM 工程 case study，至少包含 CUTLASS/CuTe 或 DeepGEMM、FP8/FP4 scaling、JIT/architecture dispatch 和 vLLM 读码笔记。

## Module 16

提交 NCCL collective 形状分析、GPU/NIC 拓扑图、DeepEP dispatch/combine 路径图和 MoE 通信预算。

## Module 17

提交一个 PyTorch CUDA custom op，包含 schema、launcher、fake/meta 设计、correctness tests、benchmark 和 vLLM/SGLang 接入设计。

## Module 18

提交 attention reference、KV cache paging 模拟、PagedAttention/MLA 路径图和 decode attention memory budget。

## Module 19

提交 CUDA/Triton/TileLang/CUTLASS 路线对比报告，包含同一算子的实现、性能、可维护性、fallback 和框架接入结论。
