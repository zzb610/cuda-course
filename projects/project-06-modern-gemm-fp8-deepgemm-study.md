# Project 06: Modern GEMM, FP8/FP4, CUTLASS And DeepGEMM Study

Recommended after: Modules 13-15  
Estimated time: 25-45 小时

## Goal

研究现代 GEMM 工程，而不是只写一个 naive matmul。你要解释 dtype、layout、Tensor Core、epilogue、JIT、architecture dispatch 和框架接入如何共同决定一个 GEMM kernel 的真实价值。

## Tracks

选择一条主线：

- CUTLASS/CuTe matmul case study。
- DeepGEMM FP8/FP4/BF16 case study。若要实跑 upstream DeepGEMM，先核对当前 README 的 SM90/SM100、CUDA、PyTorch 和编译器要求；没有匹配 GPU 时改做读码、接口和 benchmark plan。
- vLLM 中 GEMM/quantization/MoE kernel dispatch 读码报告。

如果没有合适 GPU，可以完成读码、图示和 benchmark plan，但必须说明无法实测的原因。

## Required Deliverables

- GEMM problem shape matrix。
- DType and accumulation policy。
- Layout and scaling factor explanation。
- Tensor Core path explanation。
- Architecture support matrix。
- Baseline comparison plan: cuBLAS, PyTorch, CUTLASS or DeepGEMM。
- Correctness tolerance policy。
- Profiler or source-level bottleneck analysis。
- vLLM/SGLang integration note。

## Questions To Answer

1. 这个 GEMM 是 dense、batched、grouped 还是 MoE-style？
2. Tensor Core instruction tile 如何影响更高层 tile？
3. FP8/FP4 scaling factor 放在哪里？
4. epilogue 做了什么？
5. JIT 或 architecture dispatch 解决什么问题？
6. 如果换成 SM80、SM90、SM100，哪些结论会变化？

## Expert Bar

最终报告要让读者知道：这个 GEMM 实现为什么存在，它和 cuBLAS/PyTorch baseline 的关系是什么，它对硬件代际有什么依赖，以及它如何进入推理框架。
