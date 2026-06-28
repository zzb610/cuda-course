# CUDA Course Rubric

## Mastery Levels

Beginner:

- 能构建和运行 CUDA 程序。
- 能写简单 kernel、indexing 和 bounds check。
- 能做基本错误检查和 CUDA event timing。

Intermediate:

- 能处理 1D/2D 数据和常见边界。
- 能使用 shared memory、atomics、streams。
- 能用 profiler 找出可能瓶颈。
- 能和库实现比较。

Advanced:

- 能实现 reduction、scan、stencil、transpose、tiled matrix operation。
- 能用 Nsight Systems 和 Nsight Compute 支持优化决策。
- 能讨论 coalescing、occupancy、divergence、register pressure。

Expert:

- 能做 architecture-aware 但不绝对化的优化判断。
- 能务实选择 library、custom kernel 和工程接口。
- 能交付可测试、可 benchmark、可解释的 CUDA 项目。
- 能解释 Tensor Core、PTX/SASS、TMA/WGMMA、NCCL/RDMA 与 GPU 代际能力的关系。
- 能把算子接入 PyTorch/vLLM/SGLang，并设计 fake/meta、fallback、精度和性能验证。
- 能比较 CUDA/CUTLASS/Triton/TileLang/DeepGEMM 的工程边界。

## Project Scoring

- Correctness: 30%
- Measurement and profiling: 20%
- Optimization reasoning: 20%
- Code quality: 15%
- Explanation and source grounding: 15%

## Feedback Order

1. Correctness risks。
2. Measurement problems。
3. Performance opportunities。
4. Code quality。
5. Learning next steps。

## Minimum Standard

任何项目如果没有 correctness test，不能通过。任何性能结论如果没有硬件、输入规模、版本和测量方法，不能通过。

高级项目如果没有 source grounding，也不能通过。涉及 PyTorch/vLLM/SGLang、NCCL/RDMA、DeepGEMM、Triton、TileLang 的结论必须说明参考版本、链接或本地源码路径。
